import hashlib
import json
from typing import Optional, Tuple, List

from fastapi import Depends, APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pymongo import DESCENDING
from starlette import status

from api.api_models.document import DocumentInDB, DocumentStatus
from api.api_models.lightbulb_score import LightbulbScores
from api.fake_score import predict_attribute
from api.server_config import API_ATTRIBUTES_COLLECTION_NAME, API_DEBUG, API_MONGODB_DB_NAME, API_HISTOGRAMS_PATH, \
    API_DEBUG_USER_ID, API_MOST_IMPORTANT_ATTRIBUTES, API_ADMIN_USER_ID
from api.server_dao.analysis import DAOAsyncAnalysis
from api.server_dao.document import DAOAsyncDocument
from api.api_models.analysis import AnalysisInDB, AnalysisData, AnalysisType, AnalysisStatus
from api.api_models.request import LightbulbScoreRequestData
from api.api_models.response import BackgroundTaskStatusResponse, AnalysisResultsResponse, \
    LightbulbScoreResponse, DocumentPreprocessingStillRunningResponse, DocumentPreprocessingFinishedResponse, \
    DocumentDataWithAnalyses, DocumentLevelAnalysis, ChunkLevelAnalysis, ChunkLevelSubanalysis, \
    UserDocumentsWithAnalyses, DocumentLevelAnalysisAdditionalDetails, ChunkLevelAnalysisAdditionalDetails, \
    ChunkLevelSubanalysisAdditionalDetails, DocumentWithAnalysesAdditionalDetails, HistogramDataWithMetadata, \
    AllHistogramsDTO, DocumentPreprocessingFailedResponse, DocumentFakeScore
from api.analyser import compare_2_hists, compute_histogram_data, is_attribute_available_in_partial_attributes
from api.security import verify_token
from api.server_dao.lightbulb_score import DAOAsyncLightbulbScore
from api.utils import _validate_analysis, _handle_analysis_status, calculate_lightbulb_scores, \
    get_precompiled_lightbulb_scores

from dao.attribute import DAOAsyncAttributePL
from models.attribute import AttributePLInDB

router = APIRouter()
dao_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_document: DAOAsyncDocument = DAOAsyncDocument()
dao_attribute: DAOAsyncAttributePL = DAOAsyncAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME,
                                                         db_name=API_MONGODB_DB_NAME)


@router.get("/document-preprocessing-status",
            response_model=DocumentPreprocessingStillRunningResponse | DocumentPreprocessingFinishedResponse,
            status_code=status.HTTP_200_OK)
async def document_preprocessing_status(document_hash: str,
                                        user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    document: Optional[DocumentInDB] = await dao_document.find_one_by_query(
        {'document_hash': document_hash, 'owner_id': user_id})
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )

    if document.document_status == DocumentStatus.PREPROCESS_RUNNING:
        return DocumentPreprocessingStillRunningResponse()
    elif document.document_status == DocumentStatus.READY_FOR_ANALYSIS:
        return DocumentPreprocessingFinishedResponse()
    elif document.document_status == DocumentStatus.FAILED:
        return DocumentPreprocessingFailedResponse()
    else:
        raise HTTPException(
            status_code=500,
            detail="Error while checking document preprocessing status, please try again later"
        )


@router.get("/document-analysis-status",
            response_model=BackgroundTaskStatusResponse,
            status_code=status.HTTP_200_OK)
async def document_analysis_status(analysis_id: str,
                                   _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    analysis: Optional[AnalysisInDB] = await dao_analysis.find_one_by_query({'analysis_id': analysis_id})
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis with the specified hash does not exist"
        )

    return await _handle_analysis_status(analysis)


@router.get("/document-analysis-result",
            response_model=AnalysisResultsResponse | BackgroundTaskStatusResponse,
            status_code=status.HTTP_200_OK)
async def document_analysis_results(analysis_id: str,
                                    _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    validation_result = await _validate_analysis(analysis_id)
    if isinstance(validation_result, Tuple):
        analysis, attribute = validation_result
    else:
        return validation_result

    analysis_data: AnalysisData = AnalysisData.from_analysis_and_attribute(analysis, attribute)

    return AnalysisResultsResponse(
        analysis_data=analysis_data
    )


# this is closer to get endpoint, however, it is a post endpoint because it requires a body with attribute names
@router.post("/lightbulbs-scores",
             response_model=LightbulbScoreResponse,
             status_code=status.HTTP_200_OK)
async def lightbulb_score(analysis_id: str, request_data: LightbulbScoreRequestData,
                          _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    validation_result = await _validate_analysis(analysis_id)
    if isinstance(validation_result, Tuple):
        analysis, attribute = validation_result
    else:
        return validation_result
    attribute_names = request_data.attribute_names
    lightbulb_score_data = calculate_lightbulb_scores(attribute, attribute_names)

    return LightbulbScoreResponse(
        lightbulb_scores=lightbulb_score_data
    )


@router.get("/histogram-image",
            status_code=status.HTTP_200_OK)
async def get_graph_image(analysis_id: str, attribute_name: str,
                          _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    validation_result = await _validate_analysis(analysis_id)
    if isinstance(validation_result, Tuple):
        analysis, attribute = validation_result
    else:
        return validation_result

    attribute_dict = attribute.to_flat_dict_normalized()
    if attribute_name not in attribute_dict:
        raise HTTPException(
            status_code=404,
            detail="No attributes found connected with the analysis"
        )

    compare_2_hists(attribute_name=attribute_name, file_name=f"{analysis_id}_{attribute_name}",
                    additional_value=attribute_dict[attribute_name])
    image_path = f"{API_HISTOGRAMS_PATH}/{analysis_id}_{attribute_name}.png"
    return FileResponse(image_path, media_type="image/png")

@router.get("/histogram-data", response_model=HistogramDataWithMetadata, status_code=status.HTTP_200_OK)
async def get_graph_summary(
    attribute_name: str,
    num_bins: int = Query(21, gt=1, le=100),
    min_value: Optional[float] = Query(None),
    max_value: Optional[float] = Query(None),
    existing_hash: Optional[str] = Query(None, alias="hash"),
    is_partial_attribute: Optional[bool] = Query(False),
    _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID
):
    try:
        histogram_data= await _get_graph_summary(
            analysis_id=None,
            attribute_name=attribute_name,
            num_bins=num_bins,
            min_value=min_value,
            max_value=max_value,
            existing_hash=existing_hash,
            is_partial_attribute=is_partial_attribute
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=f"No reference data for {attribute_name} found"
        )

    return HistogramDataWithMetadata(
        histogram_data=histogram_data,
        attribute_name=attribute_name,
        is_partial_data=is_partial_attribute)

@router.get("/all-histograms-data", response_model=AllHistogramsDTO, status_code=status.HTTP_200_OK)
async def get_all_graph_summary(
    num_bins: int = Query(21, gt=1, le=100),
    existing_hash: Optional[str] = Query(None, alias="hash"),
    is_partial_attribute: Optional[bool] = Query(False),
    _: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID
):
    histograms_data_with_metadata: List[HistogramDataWithMetadata] = []
    for attribute_name in API_MOST_IMPORTANT_ATTRIBUTES:
        if not is_partial_attribute:
            histogram_data = await _get_graph_summary(
                analysis_id=None,
                attribute_name=attribute_name,
                num_bins=num_bins,
                min_value=None,
                max_value=None,
                existing_hash=None,
                is_partial_attribute=False
            )
            histograms_data_with_metadata.append(HistogramDataWithMetadata(
                histogram_data = histogram_data,
                attribute_name=attribute_name,
                is_partial_data=False)
            )
        else:
            if is_attribute_available_in_partial_attributes(attribute_name):
                partial_histogram_data = await _get_graph_summary(
                    analysis_id=None,
                    attribute_name=attribute_name,
                    num_bins=num_bins,
                    min_value=None,
                    max_value=None,
                    existing_hash=None,
                    is_partial_attribute=True
                )
                histograms_data_with_metadata.append(HistogramDataWithMetadata(
                    histogram_data=partial_histogram_data,
                    attribute_name=attribute_name,
                    is_partial_data=True)
                )

    all_histograms_dto = AllHistogramsDTO(
        histograms_data_with_metadata=histograms_data_with_metadata,
        object_hash=""
    )
    all_histograms_dto.object_hash = all_histograms_dto.calculate_all_histogram_hash()
    if existing_hash and existing_hash == all_histograms_dto.object_hash:
        return JSONResponse(content={"detail": "Object hash did not change"}, status_code=200)

    return all_histograms_dto


async def _get_graph_summary(
    analysis_id: Optional[str],
    attribute_name: str,
    num_bins: int,
    min_value: Optional[float],
    max_value: Optional[float],
    existing_hash: Optional[str],
    is_partial_attribute: bool=False
):
    # validation_result = await _validate_analysis(analysis_id)
    # if isinstance(validation_result, Tuple):
    #     analysis, attribute = validation_result
    # else:
    #     return validation_result

    # attribute_dict = attribute.to_flat_dict_normalized()
    # if attribute_name not in attribute_dict:
    #     raise HTTPException(
    #         status_code=404,
    #         detail="No attributes found connected with the analysis"
    #     )

    dto = compute_histogram_data(
        attribute_name=attribute_name,
        num_bin=num_bins,
        min_value=min_value,
        max_value=max_value,
        additional_value=None, #attribute_dict[attribute_name],
        normalize=True,
        is_partial_attribute=is_partial_attribute
    )

    if existing_hash and existing_hash == dto.object_hash:
        return JSONResponse(content={"detail": "Object hash did not change"}, status_code=200)

    return dto

@router.get("/user-document-with-analyses-overview",
            response_model=DocumentDataWithAnalyses,
            status_code=status.HTTP_200_OK)
async def get_document_with_analyses_overview(document_hash: str, user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    return await _get_document_with_analyses_overview(document_hash=document_hash, user_id=user_id)

@router.get("/user-documents-with-analyses-overview",
            response_model=UserDocumentsWithAnalyses,
            status_code=status.HTTP_200_OK)
async def get_documents_with_analyses_overview(start_index: int = 0, limit: Optional[int] = None, user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    if user_id == API_ADMIN_USER_ID:
        query = {}
    else:
        query = {'owner_id': user_id}
    document_hashes: List[str] = await dao_document.find_document_hash_by_query_paginated(query, start_index, limit, sort={'updated_at': DESCENDING})
    documents_with_analyses: List[DocumentDataWithAnalyses] = []
    for document_hash in document_hashes:
        try:
            document_data_with_analyses = await _get_document_with_analyses_overview(document_hash=document_hash, user_id=user_id)
            documents_with_analyses.append(document_data_with_analyses)
        except HTTPException as e:
            if e.status_code == 404:
                continue

    # get hash of the documents with analyses
    json_str = json.dumps([obj.dict() for obj in documents_with_analyses], sort_keys=True)
    owned_data_hash = hashlib.sha256(json_str.encode()).hexdigest()

    return UserDocumentsWithAnalyses(
        documents_with_analyses=documents_with_analyses,
        owned_data_hash=owned_data_hash
    )


@router.get("/user-document-with-analyses-details",
            response_model=DocumentWithAnalysesAdditionalDetails,
            status_code=status.HTTP_200_OK)
async def get_user_document_with_analyses_details(document_hash: str, user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    if user_id == API_ADMIN_USER_ID:
        # we need just one of a document with a specified hash so just ignore user_id and continue
        query = {'document_hash': document_hash}
    else:
        query = {'document_hash': document_hash, 'owner_id': user_id}
    document: Optional[DocumentInDB] = await dao_document.find_one_by_query(query)
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )
    document_level_analysis_details = DocumentLevelAnalysisAdditionalDetails(analysed_text=document.plaintext_content_preprocessed if document.plaintext_content_preprocessed else document.plaintext_content)
    chunk_analyses: list[AnalysisInDB] = await dao_analysis.find_many_by_query(
        {'document_hash': document.document_hash, 'type': AnalysisType.CHUNK_LEVEL, 'status': AnalysisStatus.FINISHED})
    if len(chunk_analyses) == 0:
        chunk_level_analysis_details = ChunkLevelAnalysisAdditionalDetails(subanalyses_details=[])
    else:
        subanalyses_details = []
        attribute_id = chunk_analyses[0].attributes_id
        attribute: AttributePLInDB = await dao_attribute.find_by_id(attribute_id)
        if not attribute:
            raise HTTPException(
                status_code=404,
                detail="Attribute for the specified analysis does not exist"
            )
        for chunk_attributes in attribute.partial_attributes:
            subanalyses_details.append(ChunkLevelSubanalysisAdditionalDetails(
                identifier=chunk_attributes.index,
                analysed_text=chunk_attributes.partial_text
            ))
        chunk_level_analysis_details = ChunkLevelAnalysisAdditionalDetails(subanalyses_details=subanalyses_details)

    return DocumentWithAnalysesAdditionalDetails(
        document_hash=document.document_hash,
        document_level_analysis_details=document_level_analysis_details,
        chunk_level_analysis_details=chunk_level_analysis_details,
    )

dao_async_lightbulb: DAOAsyncLightbulbScore = DAOAsyncLightbulbScore()
async def _get_document_with_analyses_overview(document_hash: str, user_id: str):
    if user_id == API_ADMIN_USER_ID:
        # we need just one of a document with a specified hash so just ignore user_id and continue
        query = {'document_hash': document_hash}
    else:
        query = {'document_hash': document_hash, 'owner_id': user_id}
    document: Optional[DocumentInDB] = await dao_document.find_one_by_query(query,
                                                                  projections=['document_hash', 'document_status', 'document_name', 'created_at'])
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )

    newest_analyses: Optional[AnalysisInDB] = await dao_analysis.find_one_by_query({'document_hash': document.document_hash, 'type': AnalysisType.DOCUMENT_LEVEL, 'status': AnalysisStatus.FINISHED},
                                                                                   sort=[("start_time", -1)])
    if newest_analyses is None:
        all_analyses: List[AnalysisInDB] = await dao_analysis.find_many_by_query({'document_hash': document.document_hash})
        highest_document_level_analysis_status = AnalysisStatus.NOT_REQUESTED
        highest_chunk_level_analysis_status = AnalysisStatus.NOT_REQUESTED
        for analysis in all_analyses:
            if analysis.type == AnalysisType.DOCUMENT_LEVEL:
                if analysis.status > highest_document_level_analysis_status:
                    highest_document_level_analysis_status = analysis.status
            elif analysis.type == AnalysisType.CHUNK_LEVEL:
                if analysis.status > highest_chunk_level_analysis_status:
                    highest_chunk_level_analysis_status = analysis.status


        return DocumentDataWithAnalyses(
            document_hash=document.document_hash,
            document_status=document.document_status,
            document_name=document.document_name,
            document_upload_date=document.created_at.isoformat(),
            document_level_analysis=DocumentLevelAnalysis(
                status=highest_document_level_analysis_status,
                lightbulb_features=[]
            ),
            chunk_level_analyses=ChunkLevelAnalysis(
                status=highest_chunk_level_analysis_status,
                subanalyses=[]
            ),
            document_fake_score=None
        )

    attribute: AttributePLInDB = await dao_attribute.find_by_id(newest_analyses.attributes_id)
    if not attribute:
        raise HTTPException(
            status_code=404,
            detail="Attribute for the specified analysis does not exist"
        )

    lightbulb_features, attributes_names_left = await get_precompiled_lightbulb_scores(attribute, API_MOST_IMPORTANT_ATTRIBUTES)
    if len(attributes_names_left) > 0:
        is_only_combination_features = True
        for attribute_name in attributes_names_left:
            if 'combination_features' not in attribute_name:
                is_only_combination_features = False

        if not is_only_combination_features:
            lightbulb_features_left = calculate_lightbulb_scores(attribute, attributes_names_left)
            lightbulb_features += lightbulb_features_left
            # some of the lightbulbs where not precalculated, we should update the cached lightbulbs in db
            lightbulb_scores_model = LightbulbScores(attribute_id=attribute.id,
                                                     lightbulb_scores_dict={lightbulb.attribute_name: lightbulb for
                                                                            lightbulb in
                                                                            lightbulb_features})
            await dao_async_lightbulb.delete_one({'attribute_id': attribute.id, 'is_chunk_attribute': False})
            await dao_async_lightbulb.insert_one(lightbulb_scores_model)

    document_level_analysis = DocumentLevelAnalysis(
        status=AnalysisStatus.FINISHED, # it has to be finished as we are fetching only finished analyses
        lightbulb_features= lightbulb_features
    )

    chunk_analyses: list[AnalysisInDB] = await dao_analysis.find_many_by_query({'document_hash': document.document_hash,
                                                                                'type': AnalysisType.CHUNK_LEVEL, 'status': AnalysisStatus.FINISHED})
    if len(chunk_analyses) == 0:
        all_analyses: List[AnalysisInDB] = await dao_analysis.find_many_by_query({'document_hash': document.document_hash,
                                                                                'type': AnalysisType.CHUNK_LEVEL})
        highest_chunk_level_analysis_status = AnalysisStatus.NOT_REQUESTED
        for analysis in all_analyses:
            if analysis.status > highest_chunk_level_analysis_status:
                highest_chunk_level_analysis_status = analysis.status
        chunk_level_analysis = ChunkLevelAnalysis(
            status=highest_chunk_level_analysis_status,
            subanalyses=[]
        )
    else:
        # We need to download the attribute again as the chunk analysis might have finished after our last download
        attribute: AttributePLInDB = await dao_attribute.find_by_id(newest_analyses.attributes_id)
        chunk_level_subanalyses: list[ChunkLevelSubanalysis] = []
        if attribute.partial_attributes is not None:
            for chunk_attributes in attribute.partial_attributes:
                identifier = chunk_attributes.index
                chunk_lightbulb_features, attributes_names_left = await get_precompiled_lightbulb_scores(chunk_attributes.attribute,
                                                                                                   API_MOST_IMPORTANT_ATTRIBUTES,
                                                                                                   is_chunk_attribute=True,
                                                                                                   attribute_id=newest_analyses.attributes_id,
                                                                                                   identifier=identifier)
                if len(attributes_names_left) > 0:
                    is_only_combination_features = True
                    for attribute_name in attributes_names_left:
                        if 'combination_features' not in attribute_name:
                            is_only_combination_features = False

                    if not is_only_combination_features:
                        lightbulb_features_left = calculate_lightbulb_scores(attribute, attributes_names_left)
                        chunk_lightbulb_features += lightbulb_features_left
                        # some of the lightbulbs where not precalculated, we should update the cached lightbulbs in db
                        lightbulb_scores_model_partial = LightbulbScores(attribute_id=attribute.id,
                                                                         is_chunk_attribute=True,
                                                                         identifier=identifier,
                                                                         lightbulb_scores_dict={
                                                                             lightbulb.attribute_name: lightbulb for
                                                                             lightbulb in chunk_lightbulb_features})
                        await dao_async_lightbulb.delete_one(
                            {'attribute_id': attribute.id, 'is_chunk_attribute': True})
                        await dao_async_lightbulb.insert_one(lightbulb_scores_model_partial)

                chunk_level_subanalyses.append(ChunkLevelSubanalysis(
                    identifier=identifier,
                    lightbulb_features=lightbulb_features
                ))
        chunk_level_analysis = ChunkLevelAnalysis(
            status=AnalysisStatus.FINISHED,
            subanalyses=chunk_level_subanalyses
        )

    document_fake_score: Optional[DocumentFakeScore] = predict_attribute(attribute) if attribute is not None else None
    return DocumentDataWithAnalyses(
        document_hash=document.document_hash,
        document_status=document.document_status,
        document_name=document.document_name,
        document_upload_date=document.created_at.isoformat(),
        document_level_analysis=document_level_analysis,
        chunk_level_analyses=chunk_level_analysis,
        document_fake_score=document_fake_score
    )
