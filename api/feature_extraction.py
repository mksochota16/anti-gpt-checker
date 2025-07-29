import asyncio
import traceback
import hashlib
from datetime import datetime
from typing import Optional

from fastapi import BackgroundTasks, Depends, APIRouter, HTTPException, status

from analysis.attribute_retriving import perform_full_analysis, perform_partial_analysis_independently
from analysis.nlp_transformations import preprocess_text
from api.server_config import API_ATTRIBUTES_COLLECTION_NAME, API_DOCUMENTS_COLLECTION_NAME, API_DEBUG, \
    API_MONGODB_DB_NAME, API_DEBUG_USER_ID, ANALYSIS_TASK_QUEUE
from api.server_dao.analysis import DAOAsyncAnalysis, DAOAnalysis
from api.server_dao.document import DAOAsyncDocument, DAODocument
from api.api_models.analysis import Analysis, AnalysisType, AnalysisStatus, AnalysisInDB
from api.api_models.document import DocumentInDB, Document, DocumentStatus
from api.api_models.request import PreprocessedDocumentRequestData
from api.security import verify_token
from dao.attribute import DAOAttributePL, DAOAsyncAttributePL
from models.attribute import AttributePL, AttributePLInDB
from models.base_mongo_model import MongoObjectId

router = APIRouter()

dao_async_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_async_document: DAOAsyncDocument = DAOAsyncDocument()


@router.post("/add-document",
             response_model=dict,
             status_code=status.HTTP_201_CREATED
             )
async def post_document(preprocessed_document: PreprocessedDocumentRequestData,
                        user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    # Check if the document already exists
    existing_doc: Optional[DocumentInDB] = await dao_async_document.find_one_by_query(
        {"document_hash": preprocessed_document.document_hash, "owner_id": user_id})
    if existing_doc:
        raise HTTPException(
            status_code=409,
            detail="Document with the specified hash already exists, please use a different ID"
        )
    else:
        if preprocessed_document.document_status:
            document_status = preprocessed_document.document_status
        else:
            document_status = DocumentStatus.READY_FOR_ANALYSIS if preprocessed_document.plaintext_content is not None else DocumentStatus.PREPROCESS_RUNNING
        document = Document(
            document_name=preprocessed_document.document_name,
            document_status=document_status,
            document_hash=preprocessed_document.document_hash,
            plaintext_content=preprocessed_document.plaintext_content,
            filepath=preprocessed_document.filepath,
            owner_id=user_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await dao_async_document.insert_one(document)
    return {"message": f"Document with name {preprocessed_document.document_name} has been inserted"}


@router.patch("/update-document",
              response_model=dict,
              status_code=status.HTTP_200_OK
              )
async def update_document(preprocessed_document: PreprocessedDocumentRequestData,
                          user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    # Check if the document already exists
    existing_doc: Optional[DocumentInDB] = await dao_async_document.find_one_by_query(
        {"document_hash": preprocessed_document.document_hash, "owner_id": user_id})
    if not existing_doc:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )
    else:
        set_fields = {}
        if existing_doc.plaintext_content:
            set_fields['document_status'] = DocumentStatus.READY_FOR_ANALYSIS
        else:
            set_fields['document_status'] = DocumentStatus.PREPROCESS_RUNNING


        for field in preprocessed_document.dict():
            if preprocessed_document.dict()[field] is not None:
                set_fields[field] = preprocessed_document.dict()[field]
                if field == 'preprocessed_content':
                    set_fields['document_status'] = DocumentStatus.READY_FOR_ANALYSIS

        set_fields['updated_at'] = datetime.now()

        if preprocessed_document.document_status is not None:
            set_fields['document_status'] = preprocessed_document.document_status

        await dao_async_document.update_one({"document_hash": preprocessed_document.document_hash, "owner_id": user_id},
                                            {'$set': set_fields})
    return {"message": f"Document with name {preprocessed_document.document_name} has been updated"}


@router.post("/trigger-analysis",
             response_model=dict,
             status_code=status.HTTP_202_ACCEPTED)
async def trigger_document_analysis(document_hash: str, background_tasks: BackgroundTasks,
                                    type_of_analysis: AnalysisType = AnalysisType.DOCUMENT_LEVEL,
                                    user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    existing_doc: Optional[DocumentInDB] = await dao_async_document.find_one_by_query(
        {"document_hash": document_hash, "owner_id": user_id})
    if not existing_doc:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )
    if existing_doc.document_status != DocumentStatus.READY_FOR_ANALYSIS:
        raise HTTPException(
            status_code=409,
            detail=f"Document is not ready for analysis, its status is {existing_doc.document_status}, please wait for preprocessing to finish"
        )
    if existing_doc.plaintext_content is None:
        raise HTTPException(
            status_code=500,
            detail=f"Document plaintext content is None while its status is {existing_doc.document_status}, this should not happen, please contact support"
        )
    # generate analysis_id
    current_analysis_id = hashlib.sha256(f"{document_hash}_{type_of_analysis}_{user_id}".encode()).hexdigest()
    current_analysis: Optional[AnalysisInDB] = await dao_async_analysis.find_one_by_query({'analysis_id': current_analysis_id})
    if current_analysis and current_analysis.status in [AnalysisStatus.QUEUED, AnalysisStatus.RUNNING, AnalysisStatus.FINISHED]:
        raise HTTPException(
            status_code=409,
            detail=f"Analysis with this document and specified type already exists, analysis_id: {current_analysis_id}"
        )
    other_type_of_analysis = AnalysisType.DOCUMENT_LEVEL if type_of_analysis == AnalysisType.CHUNK_LEVEL else AnalysisType.CHUNK_LEVEL
    other_type_analysis_id = hashlib.sha256(f"{document_hash}_{other_type_of_analysis}_{user_id}".encode()).hexdigest()
    other_type_analysis: Optional[AnalysisInDB] = await dao_async_analysis.find_one_by_query(
        {'analysis_id': other_type_analysis_id})
    if not other_type_analysis and type_of_analysis == AnalysisType.CHUNK_LEVEL:
        raise HTTPException(
            status_code=409,
            detail=f"DOCUMENT_LEVEL analysis for this document does not exist, please run it first"
        )
    if other_type_analysis and type_of_analysis == AnalysisType.CHUNK_LEVEL and other_type_analysis.status != AnalysisStatus.FINISHED:
        raise HTTPException(
            status_code=409,
            detail=f"DOCUMENT_LEVEL analysis for this document is not finished yet, please wait for it to complete"
        )

    attributes_id = None
    if other_type_analysis and type_of_analysis == AnalysisType.CHUNK_LEVEL:
        attributes_id = other_type_analysis.attributes_id

    if current_analysis and current_analysis.status == AnalysisStatus.FAILED:
        await dao_async_analysis.update_one({'analysis_id': current_analysis_id}, {'$set':
                                                                   {'type': type_of_analysis,
                                                                    'status': AnalysisStatus.QUEUED,
                                                                    'document_hash': document_hash,
                                                                    'estimated_wait_time': 30,
                                                                    'start_time': datetime.now(),
                                                                    'error_message': None
                                                                    }})
    else:
        analysis = Analysis(
            analysis_id=current_analysis_id,
            type=type_of_analysis,
            status=AnalysisStatus.QUEUED,
            document_hash=document_hash,
            estimated_wait_time=30,
            start_time=datetime.now()
        )
        await dao_async_analysis.insert_one(analysis)

    task_coro = _perform_analysis(current_analysis_id, document_hash, user_id, type_of_analysis, attributes_id)
    task_id = await ANALYSIS_TASK_QUEUE.enqueue(task_coro)
    pos = ANALYSIS_TASK_QUEUE.get_position(task_id)
    await dao_async_analysis.update_one({'analysis_id': current_analysis_id}, {'$set': {'task_id': str(task_id), 'queue_position': pos}})


    return {"message": f"{type_of_analysis} analysis of document {document_hash} has been queued",
            "analysis_id": str(current_analysis_id),
            "place_in_queue": pos}


dao_async_attribute: DAOAsyncAttributePL = DAOAsyncAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME,
                                               db_name=API_MONGODB_DB_NAME)

async def _perform_analysis(analysis_id: str, document_hash, user_id: str, type_of_analysis: AnalysisType, document_level_attributes_id: Optional[MongoObjectId]):
    await dao_async_attribute.update_one({'analysis_id': analysis_id},
                                        {'$set': {'status': AnalysisStatus.RUNNING, 'queue_position': 0}})

    document: DocumentInDB = await dao_async_attribute.find_one_by_query({'document_hash': document_hash, 'owner_id': user_id})
    try:
        text_to_analyse = preprocess_text(document.plaintext_content)
        if type_of_analysis == AnalysisType.DOCUMENT_LEVEL:
            analysis_result = perform_full_analysis(text_to_analyse, 'pl', skip_partial_attributes=True)
            attribute_to_insert = AttributePL(
                referenced_db_name=API_DOCUMENTS_COLLECTION_NAME,
                referenced_doc_id=document.id,
                language="pl",
                is_generated=None,
                is_personal=None,
                **analysis_result.dict()
            )
            attributes_id = await dao_async_attribute.insert_one(attribute_to_insert)
            await dao_async_attribute.update_one({'analysis_id': analysis_id},
                                    {'$set': {'status': AnalysisStatus.FINISHED, "attributes_id": attributes_id,
                                                     'task_id': None, 'queue_position': None}})
        elif type_of_analysis == AnalysisType.CHUNK_LEVEL:
            assert document_level_attributes_id is not None, "Document level attributes ID must be provided for chunk level analysis"
            document_level_attribute_in_db: AttributePLInDB = await dao_async_attribute.find_by_id(document_level_attributes_id)
            partial_attributes, combination_features = perform_partial_analysis_independently(document_level_attribute_in_db, text_to_analyse, 'pl')
            await dao_async_attribute.update_one({'_id': document_level_attributes_id},{"$set":
                                                        {'partial_attributes': [partial_attribute.dict() for partial_attribute in partial_attributes],
                                                         'combination_features': combination_features.dict()}})
            await dao_async_attribute.update_one({'analysis_id': analysis_id},
                                    {'$set': {'status': AnalysisStatus.FINISHED, "attributes_id": document_level_attributes_id,
                                                     'task_id': None, 'queue_position': None}})


    except Exception as e:
        await dao_async_attribute.update_one({'analysis_id': analysis_id}, {'$set':
                                                                   {'status': AnalysisStatus.FAILED,
                                                                    'error_message': traceback.format_exc(),
                                                                    'task_id': None, 'queue_position': None}})
