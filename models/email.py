from enum import Enum
from typing import Union, List, Optional

from pydantic import BaseModel
from datetime import datetime

from models.base_mongo_model import MongoDBModel, MongoObjectId


class GithubClassEnums(str, Enum):
    CALENDAR = "calendar"
    PERSONAL = "personal"
    MEETINGS = "meetings"


class EmailBase(BaseModel):
    from_address: Optional[str]
    to_address: Optional[Union[str, List[str]]]
    date: Optional[datetime]
    subject: Optional[str]
    body: Optional[str]
    is_html: bool = False
    is_spam: Optional[bool]
    is_ai_generated: Optional[bool]
    detected_language: Optional[str]
    lemmatized_subject: Optional[str]
    lemmatized_body: Optional[str]
    text_plain: Optional[str]

class Email(EmailBase):
    pass

class EmailInDB(MongoDBModel, EmailBase):
    pass

class EmailGithub(EmailBase):
    inner_classification: GithubClassEnums # calendar, personal, meetings


class EmailGithubInDB(MongoDBModel, EmailGithub):
    pass


class EmailSpamAssassin(EmailBase):
    pass

class EmailSpamAssassinInDB(MongoDBModel, EmailSpamAssassin):
    pass


class EmailGmail(EmailBase):
    is_html: Optional[bool] = False
    from_name: Optional[str]
    email_labels: Optional[str]

class EmailGmailInDB(MongoDBModel, EmailGmail):
    pass


class EmailGenerated(BaseModel):
    og_db_name: str
    og_doc_id: MongoObjectId
    subject: Optional[str]
    text_plain: Optional[str]
    language: Optional[str]
    is_ai_generated: Optional[bool] = True
    placeholders_present: Optional[bool] = False
    possible_advertisement: Optional[bool] = False
    possibly_og_generated: Optional[bool] = False
    lemmatized_body: Optional[str]


class EmailGeneratedInDB(MongoDBModel, EmailGenerated):
    pass


class EmailTone(str, Enum):
    FORMAL = "formal"
    NEUTRAL = "neutral"
    INFORMAL = "informal"

class EmailInfoForGPT(BaseModel):
    summary: str
    length: int
    tone: str

    def to_prompt(self, subject: str, lang_code: str) -> str:
        match lang_code:
            case 'en':
                return f"subject: {subject}\nsummary: {self.summary}\ntone: {self.tone}\nlength: {self.length} words"
            case 'pl':
                return f"temat: {subject}\npodsumowanie: {self.summary}\nton: {self.tone}\ndługość: {self.length} słów"
            case _:
                raise ValueError(f"Language {lang_code} is not supported")

