from datetime import timedelta, datetime
from typing import Tuple, Optional, List

from fastapi import HTTPException

from api.analyser import calculate_lightbulb_score
from api.api_models.analysis import AnalysisInDB, AnalysisStatus
from api.api_models.lightbulb_score import LightbulbScoreType, LightbulbScoreData, LightbulbScoresInDB
from api.api_models.response import BackgroundTaskStatusResponse, BackgroundTaskFailedResponse, \
    BackgroundTaskRunningResponse, BackgroundTaskFinishedResponse, BackgroundTaskQueuedResponse
from api.server_config import API_ATTRIBUTES_COLLECTION_NAME, API_MONGODB_DB_NAME, API_LIGHTBULBS_SCORES_PARAMETERS
from api.server_dao.analysis import DAOAsyncAnalysis
from api.server_dao.lightbulb_score import DAOAsyncLightbulbScore
from dao.attribute import DAOAsyncAttributePL
from models.attribute import AttributePLInDB

dao_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_attribute: DAOAsyncAttributePL = DAOAsyncAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME,
                                                         db_name=API_MONGODB_DB_NAME)


async def _validate_analysis(
        analysis_id: str) -> Tuple[
                                 AnalysisInDB, AttributePLInDB] | BackgroundTaskStatusResponse:
    analysis: Optional[AnalysisInDB] = await dao_analysis.find_one_by_query({'analysis_id': analysis_id})
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis with the specified hash does not exist"
        )

    if analysis.status != AnalysisStatus.FINISHED:
        return await _handle_analysis_status(analysis)

    attribute: AttributePLInDB = await dao_attribute.find_by_id(analysis.attributes_id)
    if not attribute:
        raise HTTPException(
            status_code=404,
            detail="No attributes found connected with the analysis"
        )

    return analysis, attribute


async def _handle_analysis_status(analysis: AnalysisInDB) -> BackgroundTaskStatusResponse:
    pos = None
    from api.server_config import ANALYSIS_TASK_QUEUE
    if ANALYSIS_TASK_QUEUE is not None and analysis.task_id is not None:
        pos = ANALYSIS_TASK_QUEUE.get_position(analysis.task_id)
        await dao_analysis.update_one({'analysis_id': analysis.task_id}, {'$set': {'queue_position': pos}})

    if analysis.status == AnalysisStatus.FAILED:
        return BackgroundTaskFailedResponse(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_hash,
            estimated_wait_time=0
        )
    elif analysis.status == AnalysisStatus.QUEUED:
        return BackgroundTaskQueuedResponse(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_hash,
            place_in_queue=pos
        )
    elif analysis.status == AnalysisStatus.RUNNING:
        estimated_end_time = analysis.start_time + timedelta(seconds=analysis.estimated_wait_time)
        remaining_time = (estimated_end_time - datetime.now()).total_seconds()
        if remaining_time < 10:
            # if remaining time is less than 10 seconds, set it to 30 seconds
            remaining_time = 30
        return BackgroundTaskRunningResponse(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_hash,
            estimated_wait_time=remaining_time
        )
    elif analysis.status == AnalysisStatus.FINISHED:
        return BackgroundTaskFinishedResponse(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_hash,
            estimated_wait_time=0
        )
    else:
        raise Exception(f"Unknown analysis status: {analysis.status}")


def calculate_lightbulb_scores(attribute, attribute_names, is_chunk_attribute: bool = False) -> list[LightbulbScoreData]:
    attribute_dict = attribute.to_flat_dict_normalized()
    lightbulb_score_data = []
    for attribute_name in attribute_names:
        if attribute_name not in attribute_dict:
            continue

        if attribute_name in API_LIGHTBULBS_SCORES_PARAMETERS:
            category = API_LIGHTBULBS_SCORES_PARAMETERS[attribute_name].type
        else:  # default to BIDIRECTIONAL if not specified
            category = LightbulbScoreType.BIDIRECTIONAL

        attribute_value = attribute_dict[attribute_name]
        try:
            lightbulb_score_value = calculate_lightbulb_score(attribute_value, attribute_name,
                                                              category=category, is_chunk_attribute=is_chunk_attribute)
        except ValueError:
            continue
        lightbulb_score_data.append(LightbulbScoreData(
            attribute_name=attribute_name,
            type=category,
            score=lightbulb_score_value,
            raw_score=attribute_value,
            max_value=API_LIGHTBULBS_SCORES_PARAMETERS[attribute_name].max_value,
            min_value=API_LIGHTBULBS_SCORES_PARAMETERS[attribute_name].min_value,
            feature_rank=API_LIGHTBULBS_SCORES_PARAMETERS[attribute_name].feature_rank
        ))
    return lightbulb_score_data

dao_async_lightbulbs: DAOAsyncLightbulbScore = DAOAsyncLightbulbScore()
async def get_precompiled_lightbulb_scores(attribute: AttributePLInDB,
                                           attribute_names,
                                           is_chunk_attribute: bool = False,
                                           attribute_id: Optional[str] = None,
                                           identifier: Optional[int] = None) -> Tuple[List[LightbulbScoreData], List[str]]:
    if not is_chunk_attribute:
        lightbulbs_in_db: Optional[LightbulbScoresInDB] = await dao_async_lightbulbs.find_one_by_query(
            {'attribute_id':attribute.id, 'is_chunk_attribute':is_chunk_attribute})
    else:
        lightbulbs_in_db: Optional[LightbulbScoresInDB] = await dao_async_lightbulbs.find_one_by_query(
            {'attribute_id': attribute_id, 'is_chunk_attribute': is_chunk_attribute, 'identifier': identifier})
    if lightbulbs_in_db is None:
        return [], attribute_names

    else:
        lightbulbs_to_return = []
        attribute_names_left = []
        for attribute_name in attribute_names:
            if attribute_name in lightbulbs_in_db.lightbulb_scores_dict:
                lightbulbs_to_return.append(lightbulbs_in_db.lightbulb_scores_dict[attribute_name])
            else:
                attribute_names_left.append(attribute_name)

        return lightbulbs_to_return, attribute_names_left
