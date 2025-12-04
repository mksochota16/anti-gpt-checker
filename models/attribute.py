from __future__ import annotations

from typing import Dict, Optional, Union, List
import math

from pydantic import BaseModel

from models.base_mongo_model import MongoObjectId, MongoDBModel
from models.stylometrix_metrics import AllStyloMetrixFeaturesEN, AllStyloMetrixFeaturesPL
from models.text_errors import TextErrors
from models.combination_features import CombinationFeatures


class AttributeNoDBParameters(BaseModel):
    perplexity: Optional[float]  # V
    perplexity_base: Optional[float]  # V
    perplexity_base_normalized: Optional[float]  # V

    sample_word_counts: Optional[dict]  # V

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

    number_of_unrecognized_words_lang_tool: Optional[int]
    number_of_abbreviations_lang_tool: Optional[int]
    number_of_unrecognized_words_dict_check: Optional[int]

    combination_features: Optional[CombinationFeatures]

    partial_attributes: Optional[List[PartialAttribute]] = None

    def to_flat_dict(self):
        temp_dict = self.dict(
            exclude={"referenced_db_name", "is_generated", "is_personal", "referenced_doc_id", "language", "id",
                     "pos_eng_tags", "sentiment_eng", "lemmatized_text", "sample_word_counts"})
        flattened_dict = self._flatten_dict(temp_dict)
        return flattened_dict

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.', skip_lists: bool = False):
        items = []
        for key, value in d.items():
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, str):
                continue
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep=sep).items())
            elif isinstance(value, list):
                if skip_lists:
                    continue
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        items.extend(self._flatten_dict(item, f"{new_key}.{index}", sep=sep).items())
                    elif isinstance(item, float) or isinstance(item, int):
                        items.append((f"{new_key}.{index}", value))
                    elif isinstance(item, str):
                        continue
                    elif isinstance(item, object):
                        items.extend(self._flatten_dict(item.dict(), f"{new_key}.{index}", sep=sep).items())
                    else:
                        raise TypeError(f"Unknown type {type(item)}")
            else:
                items.append((new_key, value))
        return dict(items)

    def to_flat_dict_normalized(self, exclude=None, include_partial_attributes=False):
        temp_dict = self.dict(
            exclude={"referenced_db_name", "is_generated", "is_personal", "referenced_doc_id", "language", "id",
                     "pos_eng_tags", "sentiment_eng", "punctuation", "lemmatized_text", #"perplexity_base",
                     "sample_word_counts", "partial_attributes", "llm_model_name"})
        if exclude:
            for key in exclude:
                if key in temp_dict:
                    temp_dict.pop(key)
        flattened_dict = self._flatten_dict(temp_dict, skip_lists=True)
        flattened_dict[
            'double_spaces'] = self.double_spaces / self.number_of_characters if self.double_spaces is not None else 0
        flattened_dict[
            'no_space_after_punctuation'] = self.no_space_after_punctuation / self.number_of_characters if self.no_space_after_punctuation is not None else 0
        flattened_dict['emojis'] = self.emojis / self.number_of_characters if self.emojis is not None else 0
        flattened_dict[
            'question_marks'] = self.question_marks / self.number_of_characters if self.question_marks is not None else 0
        flattened_dict[
            'exclamation_marks'] = self.exclamation_marks / self.number_of_characters if self.exclamation_marks is not None else 0
        flattened_dict[
            'double_question_marks'] = self.double_question_marks / self.number_of_characters if self.double_question_marks is not None else 0
        flattened_dict[
            'double_exclamation_marks'] = self.double_exclamation_marks / self.number_of_characters if self.double_exclamation_marks is not None else 0
        flattened_dict[
            'number_of_errors'] = self.number_of_errors / self.number_of_characters if self.number_of_errors is not None else 0
        flattened_dict[
            'number_of_unrecognized_words_lang_tool'] = self.number_of_unrecognized_words_lang_tool / self.number_of_words if self.number_of_unrecognized_words_lang_tool is not None else 0
        flattened_dict[
            'number_of_abbreviations_lang_tool'] = self.number_of_abbreviations_lang_tool / self.number_of_words if self.number_of_abbreviations_lang_tool is not None else 0
        flattened_dict[
            'number_of_unrecognized_words_dict_check'] = self.number_of_unrecognized_words_dict_check / self.number_of_words if self.number_of_unrecognized_words_dict_check is not None else 0

        for key in flattened_dict:
            if flattened_dict[key] is None:
                flattened_dict[key] = 0
            elif (not isinstance(flattened_dict[key], list) and
                  not isinstance(flattened_dict[key], dict) and
                  math.isnan(flattened_dict[key])):
                flattened_dict[key] = 0
            elif key.startswith("text_errors_by_category."):
                flattened_dict[key] = float(flattened_dict[key]) / self.number_of_characters if flattened_dict[
                                                                                                    key] is not None else 0
            elif "partial_attribute_statistics" in key and "values" in key:
                if exclude is None:
                    exclude = []
                exclude.append(key)
            # normalization of stylometrix features is not needed as they are already normalized

        if include_partial_attributes and isinstance(self.partial_attributes, list):
            for partial_attribute in self.partial_attributes:
                flattened_partial_attribute_dict = partial_attribute.attribute.to_flat_dict_normalized()
                flattened_partial_attribute_key = f"partial_attributes.{partial_attribute.index}."
                for key, value in flattened_partial_attribute_dict.items():
                    flattened_dict[flattened_partial_attribute_key + key] = value

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
    referenced_db_name: str  # V
    referenced_doc_id: MongoObjectId  # V
    language: Optional[str]  # V
    is_generated: Optional[bool]  # V
    is_personal: Optional[bool]  # V
    llm_model_name: Optional[str]


class AttributePL(AttributeBase):
    stylometrix_metrics: Optional[AllStyloMetrixFeaturesPL]


class AttributeEN(AttributeBase):
    stylometrix_metrics: Optional[AllStyloMetrixFeaturesEN]


class AttributePLInDB(MongoDBModel, AttributePL):
    pass


class AttributeENInDB(MongoDBModel, AttributeEN):
    pass


AttributeInDB = Union[AttributePLInDB, AttributeENInDB]


class PartialAttributePL(BaseModel):
    index: int
    partial_text: str
    attribute: AttributeNoDBParametersPL


class PartialAttributeEN(BaseModel):
    index: int
    partial_text: str
    attribute: AttributeNoDBParametersEN


PartialAttribute = Union[PartialAttributePL, PartialAttributeEN]

# needed for recursive nature of PartialAttributes
AttributeNoDBParameters.update_forward_refs()
AttributeNoDBParametersPL.update_forward_refs()
AttributeNoDBParametersEN.update_forward_refs()
AttributeBase.update_forward_refs()
AttributePL.update_forward_refs()
AttributeEN.update_forward_refs()
AttributePLInDB.update_forward_refs()
AttributeENInDB.update_forward_refs()
