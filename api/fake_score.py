import pickle
from typing import Optional

from fastapi import Depends, APIRouter, HTTPException
from starlette import status

from api.api_models.document import DocumentInDB
from api.server_config import API_ATTRIBUTES_COLLECTION_NAME, API_DEBUG, API_MONGODB_DB_NAME, API_FAKE_SCORE_FEATURES, \
    API_CATBOOST_MODEL_PATH, API_DEBUG_USER_ID, API_CATBOOST_VECTORIZER
from api.server_dao.analysis import DAOAsyncAnalysis
from api.server_dao.document import DAOAsyncDocument
from api.api_models.analysis import AnalysisInDB, AnalysisType, AnalysisStatus
from api.api_models.response import DocumentFakeScore, PredictedLabel
from api.security import verify_token

from dao.attribute import DAOAsyncAttributePL
from models.attribute import AttributePLInDB

from catboost import CatBoostClassifier
router = APIRouter()
dao_analysis: DAOAsyncAnalysis = DAOAsyncAnalysis()
dao_document: DAOAsyncDocument = DAOAsyncDocument()
dao_attribute: DAOAsyncAttributePL = DAOAsyncAttributePL(collection_name=API_ATTRIBUTES_COLLECTION_NAME,
                                                         db_name=API_MONGODB_DB_NAME)

@router.get("/document-fake-score",
            response_model=DocumentFakeScore,
            status_code=status.HTTP_200_OK)
async def get_document_fake_score(document_hash: str, user_id: str = Depends(verify_token) if not API_DEBUG else API_DEBUG_USER_ID):
    document: DocumentInDB = await dao_document.find_one_by_query({'document_hash': document_hash, 'owner_id': user_id})
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document with the specified hash does not exist"
        )

    newest_analyses: Optional[AnalysisInDB] = await dao_analysis.find_one_by_query(
        {'document_hash': document.document_hash, 'type': AnalysisType.DOCUMENT_LEVEL,
         'status': AnalysisStatus.FINISHED},
        sort=[("start_time", -1)])
    if newest_analyses is None:
        raise HTTPException(
            status_code=404,
            detail="Finished analysis for this document does not exist"
        )

    attribute: Optional[AttributePLInDB] = await dao_attribute.find_by_id(newest_analyses.attributes_id)

    if attribute is None:
        raise HTTPException(
            status_code=404,
            detail="Attribute for specified analysis does not exist"
        )

    return predict_attribute(attribute)



def predict_attribute(attribute: AttributePLInDB) -> DocumentFakeScore:
    full_dict = attribute.to_flat_dict_normalized()
    filtered_dict = {key: value for key, value in full_dict.items() if key in API_FAKE_SCORE_FEATURES}


    loaded_model = CatBoostClassifier()
    loaded_model.load_model(API_CATBOOST_MODEL_PATH)
    with open(API_CATBOOST_VECTORIZER, "rb") as f:
        vectorizer = pickle.load(f)

    sample_features = vectorizer.transform([filtered_dict])
    predicted_label = loaded_model.predict(sample_features)[0]
    predicted_prob = loaded_model.predict_proba(sample_features)[0, 1] # 0-real, 1-generated

    fake_score = (predicted_prob - 0.5)*(-2) # -1-generated, 1-real
    return DocumentFakeScore(
        predicted_label=PredictedLabel.LLM_GENERATED if predicted_label==1 else PredictedLabel.HUMAN_WRITTEN,
        fake_score=fake_score
    )