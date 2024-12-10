from typing import Dict, Optional, Union
import math

from pydantic import BaseModel

from models.base_mongo_model import MongoObjectId, MongoDBModel
from models.stylometrix_metrics import AllStyloMetrixFeaturesEN, AllStyloMetrixFeaturesPL
from models.text_errors import TextErrors

class AttributeNoDBParameters(BaseModel):
    perplexity: Optional[float]  # V
    perplexity_base: Optional[float]  # V

    burstiness: Optional[float]  # V
    burstiness2: Optional[float]  # V

    average_sentence_word_length: Optional[float]  # V
    standard_deviation_sentence_word_length: Optional[float]  # V
    variance_sentence_word_length: Optional[float]  # V

    standard_deviation_sentence_char_length: Optional[float]  # V
    variance_sentence_char_length: Optional[float]  # V
    average_sentence_char_length: Optional[float]  # V

    standard_deviation_word_char_length: Optional[float]  # V
    variance_word_char_length: Optional[float]  # V
    average_word_char_length: Optional[float]  # V

    punctuation: Optional[int]  # V
    punctuation_per_sentence: Optional[float]  # P
    punctuation_density: Optional[float]  # P

    number_of_sentences: Optional[int]  # V
    number_of_words: Optional[int]  # V
    number_of_characters: Optional[int]  # V

    double_spaces: Optional[int | dict]  # V
    no_space_after_punctuation: Optional[int]  # V
    emojis: Optional[int]  # V
    question_marks: Optional[int]  # V
    exclamation_marks: Optional[int]  # V
    double_question_marks: Optional[int]  # V
    double_exclamation_marks: Optional[int]  # V

    text_errors_by_category: Optional[TextErrors]  # V
    number_of_errors: Optional[int]  # V

    lemmatized_text: Optional[str]

    pos_eng_tags: Optional[Dict[str, int]]
    sentiment_eng: Optional[Dict[str, float]]

    def to_flat_dict(self):
        temp_dict = self.dict(exclude={"referenced_db_name", "is_generated", "is_personal", "referenced_doc_id", "language", "id", "pos_eng_tags", "sentiment_eng", "lemmatized_text"})
        flattened_dict = self._flatten_dict(temp_dict)
        return flattened_dict

    def _flatten_dict(self, d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


    def to_flat_dict_normalized(self, exclude=None):
        temp_dict = self.dict(exclude={"referenced_db_name", "is_generated", "is_personal", "referenced_doc_id", "language", "id", "pos_eng_tags", "sentiment_eng", "punctuation", "perplexity_base", "lemmatized_text"})
        if exclude:
            for key in exclude:
                if key in temp_dict:
                    temp_dict.pop(key)
        flattened_dict = self._flatten_dict(temp_dict)
        flattened_dict['double_spaces'] = self.double_spaces/self.number_of_characters if self.double_spaces is not None else 0
        flattened_dict['no_space_after_punctuation'] = self.no_space_after_punctuation/self.number_of_characters if self.no_space_after_punctuation is not None else 0
        flattened_dict['emojis'] = self.emojis/self.number_of_characters if self.emojis is not None else 0
        flattened_dict['question_marks'] = self.question_marks/self.number_of_characters if self.question_marks is not None else 0
        flattened_dict['exclamation_marks'] = self.exclamation_marks/self.number_of_characters if self.exclamation_marks is not None else 0
        flattened_dict['double_question_marks'] = self.double_question_marks/self.number_of_characters if self.double_question_marks is not None else 0
        flattened_dict['double_exclamation_marks'] = self.double_exclamation_marks/self.number_of_characters if self.double_exclamation_marks is not None else 0
        flattened_dict['number_of_errors'] = self.number_of_errors/self.number_of_characters if self.number_of_errors is not None else 0

        for key in flattened_dict:
            if flattened_dict[key] is None:
                flattened_dict[key] = 0
            elif math.isnan(flattened_dict[key]):
                flattened_dict[key] = 0
            elif key.startswith("text_errors_by_category."):
                flattened_dict[key] = float(flattened_dict[key])/self.number_of_characters if flattened_dict[key] is not None else 0

            elif key.startswith("stylometrix_metrics.") and not key.startswith("stylometrix_metrics.statistics."):
                flattened_dict[key] = float(flattened_dict[key])/self.number_of_characters

        if exclude:
            for key in exclude:
                if key in flattened_dict:
                    flattened_dict.pop(key)

        return flattened_dict

class AttributeNoDBParametersPL(AttributeNoDBParameters):
    stylometrix_metrics: Optional[AllStyloMetrixFeaturesPL]

class AttributeNoDBParametersEN(AttributeNoDBParameters):
    stylometrix_metrics: Optional[AllStyloMetrixFeaturesEN]
class AttributeBase(AttributeNoDBParameters):
    referenced_db_name: str #V
    referenced_doc_id: MongoObjectId #V
    language: Optional[str] #V
    is_generated: Optional[bool] #V
    is_personal: Optional[bool] #V

class AttributePL(AttributeBase):
    stylometrix_metrics: Optional[AllStyloMetrixFeaturesPL]

class AttributeEN(AttributeBase):
    stylometrix_metrics: Optional[AllStyloMetrixFeaturesEN]
class AttributePLInDB(MongoDBModel, AttributePL):
    pass
class AttributeENInDB(MongoDBModel, AttributeEN):
    pass

AttributeInDB = Union[AttributePLInDB, AttributeENInDB]