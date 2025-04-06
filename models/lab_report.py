from typing import Optional

from pydantic import BaseModel

from models.base_mongo_model import MongoDBModel

class LabReport(BaseModel):
    plaintext_content: str
    is_generated: bool
    is_mixed: Optional[bool]
    tag: str


class LabReportInDB(MongoDBModel, LabReport):
    pass