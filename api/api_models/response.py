import hashlib
import json
from enum import Enum
from typing import Union, List, Literal, Optional

from pydantic import BaseModel
from datetime import datetime
from api.api_models.analysis import AnalysisData, AnalysisStatus, AnalysisInDB
from api.api_models.document import DocumentInDB, DocumentStatus
from api.api_models.lightbulb_score import LightbulbScoreData, LightbulbScoreType
from models.attribute import AttributeInDB


class DocumentPreprocessingStillRunningResponse(BaseModel):
    type: str = "DocumentPreprocessingStillRunningResponse"
    message: str = "Document preprocessing is still running, please wait and try again later"

class DocumentPreprocessingFailedResponse(BaseModel):
    type: str = "DocumentPreprocessingFailedResponse"
    message: str = "Document preprocessing failed, please check the document format and try again"

class DocumentPreprocessingFinishedResponse(BaseModel):
    type: str = "DocumentPreprocessingFinishedResponse"
    message: str = "Document preprocessing is finished, you can now analyze the document"


class DocumentWithSpecifiedHashAlreadyExists(BaseModel):
    type: str = "document_with_specified_hash_already_exists"
    message: str = "Document with the specified hash already exists, please use a different ID"


class BackgroundTaskFailedResponse(BaseModel):
    type: str = "failed_to_extract_data_response"
    message: str = "Failed to extract attributes from the given document"
    status: AnalysisStatus = AnalysisStatus.FAILED
    document_id: str
    estimated_wait_time: int = 0
    analysis_id: str

class BackgroundTaskQueuedResponse(BaseModel):
    type: str = "background_task_queued_response"
    message: str = "Background task is queued, please wait until other tasks in queue are finished and call document-analysis-result endpoint with the given analysis_id"
    status: AnalysisStatus = AnalysisStatus.QUEUED
    document_id: str
    place_in_queue: int
    analysis_id: str

class BackgroundTaskRunningResponse(BaseModel):
    type: str = "background_task_running_response"
    message: str = "Background task is running, please wait the given wait-time and call document-analysis-result endpoint with the given analysis_id"
    status: AnalysisStatus = AnalysisStatus.RUNNING
    document_id: str
    estimated_wait_time: int
    analysis_id: str


class BackgroundTaskFinishedResponse(BaseModel):
    type: str = "background_task_finished_response"
    message: str = "Background task is finished"
    status: AnalysisStatus = AnalysisStatus.FINISHED
    document_id: str
    estimated_wait_time: int = 0
    analysis_id: str

BackgroundTaskStatusResponse = Union[BackgroundTaskRunningResponse, BackgroundTaskFinishedResponse, BackgroundTaskFailedResponse, BackgroundTaskQueuedResponse]

class AnalysisResultsResponse(BaseModel):
    type: str = "analysis_results_response"
    message: str = "Analysis results"
    analysis_data: AnalysisData


class LightbulbScoreResponse(BaseModel):
    type: str = "lightbulb_score_response"
    lightbulb_scores: List[LightbulbScoreData]

class DocumentsOfUserResponse(BaseModel):
    type: str = "documents_of_user_response"
    message: str = "Documents of the user"
    documents: List[DocumentInDB]  # List of documents

class DocumentDeletedResponse(BaseModel):
    type: str = "document_deleted_response"
    message: str = "Document deleted"

class AnalysisWithLightbulbs(BaseModel):
    analysis: AnalysisInDB  # Analysis object
    attribute_in_db: AttributeInDB
    lightbulb_scores: List[LightbulbScoreData]  # Lightbulb scores associated with the analysis

class DocumentWithAnalysis(BaseModel):
    document: DocumentInDB  # Document object
    analyses_with_lightbulbs: List[AnalysisWithLightbulbs]

class DocumentsOfUserWithAnalysisResponse(BaseModel):
    type: str = "documents_of_user_with_analysis_response"
    message: str = "Documents of the user with analyses"
    documents_with_analyses: List[DocumentWithAnalysis]  # List of documents

class AnalysesOfDocumentsResponse(BaseModel):
    type: str = "analyses_of_documents_response"
    message: str = "Analyses of the documents"
    analyses: List[AnalysisData]  # List of AnalysisData objects


class HistogramData(BaseModel):
    feature: str
    data_type: Literal["llm-generated", "human-written"]
    bins: List[float]      # bin edges
    counts: List[float]      # histogram counts per bin #FIXME change "counts" to "heights"

class HistogramDataDTO(BaseModel):
    llm: HistogramData
    human: HistogramData
    # additional_value: Optional[float] = None
    min_value: float
    max_value: float
    num_bins: int
    object_hash: str

    def calculate_histogram_hash(self) -> str:
        dto_copy = self.dict(exclude={"object_hash"})
        # Sort keys to ensure consistent hashing
        encoded = json.dumps(dto_copy, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

class HistogramDataWithMetadata(BaseModel):
    histogram_data: HistogramDataDTO
    attribute_name: str
    is_partial_data: bool

class AllHistogramsDTO(BaseModel):
    histograms_data_with_metadata: List[HistogramDataWithMetadata]
    object_hash: str
    def calculate_all_histogram_hash(self) -> str:
        hash_concatenated = "".join(
            histogram_data_with_metadata.histogram_data.object_hash for histogram_data_with_metadata in self.histograms_data_with_metadata
        )
        return hashlib.sha256(hash_concatenated.encode('utf-8')).hexdigest()

class DocumentLevelAnalysis(BaseModel):
    status: AnalysisStatus
    lightbulb_features: List[LightbulbScoreData]

class ChunkLevelSubanalysis(BaseModel):
    identifier: int  # for ordering subanalyses
    lightbulb_features: List[LightbulbScoreData]

class ChunkLevelAnalysis(BaseModel):
    status: AnalysisStatus
    subanalyses: List[ChunkLevelSubanalysis]


class PredictedLabel(str, Enum):
    HUMAN_WRITTEN = "human_written" # score [-1,0]
    LLM_GENERATED = "llm_generated" # score [0,1]

class DocumentFakeScore(BaseModel):
    predicted_label: PredictedLabel
    fake_score: float # [-1,1]

class DocumentDataWithAnalyses(BaseModel):
    document_hash: str  # the document hash
    document_status: DocumentStatus
    document_name: str
    document_upload_date: str

    document_level_analysis: Optional[DocumentLevelAnalysis] = {}
    chunk_level_analyses: Optional[ChunkLevelAnalysis] = {}

    document_fake_score: Optional[DocumentFakeScore] = None

class UserDocumentsWithAnalyses(BaseModel):
    documents_with_analyses: List[DocumentDataWithAnalyses]
    owned_data_hash: str  # to help detect changes / invalidate cache

class DocumentLevelAnalysisAdditionalDetails(BaseModel):
    analysed_text: str

class ChunkLevelSubanalysisAdditionalDetails(BaseModel):
    identifier: int
    analysed_text: str

class ChunkLevelAnalysisAdditionalDetails(BaseModel):
    subanalyses_details: List[ChunkLevelSubanalysisAdditionalDetails]

class DocumentWithAnalysesAdditionalDetails(BaseModel):
    document_hash: str
    document_level_analysis_details: DocumentLevelAnalysisAdditionalDetails
    chunk_level_analysis_details: ChunkLevelAnalysisAdditionalDetails

class UpdatedDocumentsOfUserWithAnalysisResponse(BaseModel):
    documents_with_analyses: list[DocumentWithAnalysis]
    new_timestamp: datetime
