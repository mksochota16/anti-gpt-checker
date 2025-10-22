from typing import Optional

from pydantic import BaseModel
from numpy import std, var, mean

from models.stylometrix_metrics import AllStyloMetrixFeaturesPL, AllStyloMetrixFeaturesEN


def init_dict_from_key_and_values(values: list[float]) -> dict:
    values = values
    if len(values) == 0:
        raise ValueError("Values list cannot be empty")
    elif len(values) == 1:
        std_dev = 0
        variance = 0
        average = values[0]
    else:
        std_dev = std(values)
        variance = var(values)
        average = mean(values)

    return {
        "values":values,
        "std_dev":std_dev,
        "variance":variance,
        "average":average,
    }


class CombinationFeatures(BaseModel):
    content_function_ratio: Optional[float]
    common_long_word_ratio: Optional[float]
    common_rare_word_ratio: Optional[float]
    active_passive_voice_ratio: Optional[float]
    partial_attribute_statistics: Optional[dict]

    @staticmethod
    def init_from_stylometrix_and_partial_attributes(
            stylometrix_metrics: AllStyloMetrixFeaturesPL | AllStyloMetrixFeaturesEN,
            partial_attributes_dicts: Optional[list[dict]] = None,
            features_to_calculate: Optional[list[str]] = None):

        if features_to_calculate is None:
            features_to_calculate = [
                "combination.content_function_ratio",
                "combination.common_long_word_ratio",
                "combination.common_rare_word_ratio",
                "combination.active_passive_voice_ratio",
            ]

        content_function_ratio = None
        common_long_word_ratio = None
        common_rare_word_ratio = None
        active_passive_voice_ratio = None

        if "combination.content_function_ratio" in features_to_calculate:
            L_CONT_A = stylometrix_metrics.lexical.L_CONT_A
            L_FUNC_A = stylometrix_metrics.lexical.L_FUNC_A
            if L_CONT_A is not None and L_FUNC_A is not None:
                content_function_ratio = L_CONT_A / (L_FUNC_A or 1)

        if "combination.common_long_word_ratio" in features_to_calculate:
            L_TCCT1 = stylometrix_metrics.lexical.L_TCCT1
            L_SYL_G4 = stylometrix_metrics.lexical.L_SYL_G4
            if L_TCCT1 is not None and L_SYL_G4 is not None:
                common_long_word_ratio = L_TCCT1 / (L_SYL_G4 or 1)

        if "combination.common_rare_word_ratio" in features_to_calculate:
            L_TCCT1 = stylometrix_metrics.lexical.L_TCCT1
            L_TCCT5 = stylometrix_metrics.lexical.L_TCCT5
            if L_TCCT1 is not None and L_TCCT5 is not None:
                common_rare_word_ratio = L_TCCT1 / (1 - L_TCCT5 if L_TCCT5 else 1)

        if "combination.active_passive_voice_ratio" in features_to_calculate:
            IN_V_ACT = stylometrix_metrics.inflection.IN_V_ACT
            IN_V_PASS = stylometrix_metrics.inflection.IN_V_PASS
            if IN_V_ACT is not None and IN_V_PASS is not None:
                active_passive_voice_ratio = IN_V_ACT / (IN_V_PASS or 1)

        partial_attribute_statistics = None
        if partial_attributes_dicts:
            partial_attribute_statistics = {}
            for key in partial_attributes_dicts[0]:
                values = [p[key] for p in partial_attributes_dicts]
                if isinstance(values[0], (float, int)):
                    partial_attribute_statistics[key] = init_dict_from_key_and_values(values)

        return CombinationFeatures(
            content_function_ratio=content_function_ratio,
            common_long_word_ratio=common_long_word_ratio,
            common_rare_word_ratio=common_rare_word_ratio,
            active_passive_voice_ratio=active_passive_voice_ratio,
            partial_attribute_statistics=partial_attribute_statistics,
        )
"""
@staticmethod
    def init_from_stylometrix_and_partial_attributes(
            stylometrix_metrics: AllStyloMetrixFeaturesPL | AllStyloMetrixFeaturesEN,
            partial_attributes_dicts: Optional[list[dict]] = None):
        content_function_ratio = stylometrix_metrics.lexical.L_CONT_A / (
            stylometrix_metrics.lexical.L_FUNC_A if stylometrix_metrics.lexical.L_FUNC_A else 1)
        common_long_word_ratio = stylometrix_metrics.lexical.L_TCCT1 / (
            stylometrix_metrics.lexical.L_SYL_G4 if stylometrix_metrics.lexical.L_SYL_G4 else 1)
        common_rare_word_ratio = stylometrix_metrics.lexical.L_TCCT1 / (
            1 - stylometrix_metrics.lexical.L_TCCT5 if stylometrix_metrics.lexical.L_TCCT5 else 1)
        active_passive_voice_ratio = stylometrix_metrics.inflection.IN_V_ACT / (
            stylometrix_metrics.inflection.IN_V_PASS if stylometrix_metrics.inflection.IN_V_PASS else 1)

        if partial_attributes_dicts:
            partial_attribute_statistics = {}
            for key in partial_attributes_dicts[0]:
                values = [partial_attribute[key] for partial_attribute in partial_attributes_dicts]
                if not (isinstance(values[0], float) or isinstance(values[0], int)):
                    continue
                partial_attribute_statistics[key] = init_dict_from_key_and_values(values)
        else:
            partial_attribute_statistics = None

        return CombinationFeatures(
            content_function_ratio=content_function_ratio,
            common_long_word_ratio=common_long_word_ratio,
            common_rare_word_ratio=common_rare_word_ratio,
            active_passive_voice_ratio=active_passive_voice_ratio,
            partial_attribute_statistics=partial_attribute_statistics
        )
"""