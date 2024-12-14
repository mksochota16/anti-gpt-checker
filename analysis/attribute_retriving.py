import string
import re
import math
from typing import Dict, List, Union, Optional, Tuple

import nltk
import numpy as np
from collections import defaultdict

import pandas as pd
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import stylo_metrix as sm
from numpy import std, var, mean
from pandas import DataFrame
from collections import Counter
from langdetect import detect as langdetect_detect, DetectorFactory, LangDetectException
import langid

import pycld2 as cld2
import cld3
from textblob import TextBlob


from analysis.nlp_transformations import replace_links_with_text, remove_stopwords_punctuation_emojis_and_splittings, \
    lemmatize_text, split_into_sentences, is_abbreviation
from models.attribute import AttributeNoDBParametersPL, AttributeNoDBParametersEN

from models.stylometrix_metrics import AllStyloMetrixFeaturesEN, AllStyloMetrixFeaturesPL
import html2text

html2text_handler = html2text.HTML2Text()
MAX_STYLOMETRIX_LENGTH = 512


def average_word_length(text: str) -> float:
    """
    Calculate the average length of words in the given text.

    Parameters:
    - text (str): Input text.

    Returns:
    - float: Average word length.
    """
    words = word_tokenize(text)
    word_lengths = [len(word) for word in words]
    return sum(word_lengths) / len(word_lengths) if len(word_lengths) > 0 else 0


def average_sentence_length(text: str) -> float:
    """
    Calculate the average length of sentences in the given text.

    Parameters:
    - text (str): Input text.

    Returns:
    - float: Average sentence length.
    """
    sentences = re.split(r'[.!?]', text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    return sum(sentence_lengths) / len(sentence_lengths) if len(sentence_lengths) > 0 else 0


def count_pos_tags_eng(text: str, pos_tags=None) -> Dict[str, int]:
    """
    Count the occurrences of specified part-of-speech tags in the given English text.

    Parameters:
    - text (str): Input text.
    - pos_tags (List[str]): List of part-of-speech tags to count.

    Returns:
    - dict: Dictionary containing counts for each specified part-of-speech tag.
    """
    if pos_tags is None:
        pos_tags = ['NN', 'VB', 'JJ']

    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    pos_counts = {tag: 0 for tag in pos_tags}

    for word, tag in tagged_words:
        if tag in pos_counts:
            pos_counts[tag] += 1

    return pos_counts


def sentiment_score_eng(text: str) -> float:
    """
    Calculate the sentiment polarity score of the given English text.

    Parameters:
    - text (str): Input text.

    Returns:
    - float: Sentiment polarity score.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity


def count_punctuation(text: str) -> float:
    """
    Count the occurrences of punctuation marks in the given text.

    Parameters:
    - text (str): Input text.

    Returns:
    - float: Count of punctuation marks per length of text
    """
    punctuation_count = sum([1 for char in text if char in '.,;:!?'])
    if len(text) == 0:
        return 0
    return punctuation_count / len(text)


def stylo_metrix_analysis(texts: List[str], language_code: str) -> list[
    AllStyloMetrixFeaturesPL | AllStyloMetrixFeaturesEN]:
    stylo = sm.StyloMetrix(language_code)

    tokens = stylo.nlp(texts[0])
    if len(tokens) < MAX_STYLOMETRIX_LENGTH:
        metrics = stylo.transform(texts[0])
        converted_metrics = stylo_metrix_output_to_model(metrics, language_code)
        return converted_metrics
    else:
        converted_metrics = []
        stylometrix_analysis_slice_of_text(texts[0], language_code, converted_metrics, stylo)
        averaged_metrics = {}
        for metric in converted_metrics[0].to_dict().keys():
            if metric == "text":
                continue
            metric_values = [metric_dict.to_dict()[metric][0] for metric_dict in converted_metrics]
            averaged_metrics[metric] = sum(metric_values) / len(metric_values)

        averaged_metrics["text"] = texts[0]
        df = pd.DataFrame(averaged_metrics, index=[0])
        converted_metrics = stylo_metrix_output_to_model(df, language_code)
        return converted_metrics

def stylometrix_analysis_slice_of_text(text: str, language_code: str, already_analysed_list: list, stylo):
    tokens = stylo.nlp(text)
    if len(tokens) < MAX_STYLOMETRIX_LENGTH:
        metrics = stylo.transform(text)
        already_analysed_list.append(metrics)
    else:
        stylometrix_analysis_slice_of_text(text[:len(text) // 2], language_code, already_analysed_list, stylo)
        stylometrix_analysis_slice_of_text(text[len(text) // 2:], language_code, already_analysed_list, stylo)



def stylo_metrix_output_to_model(metrics_df: DataFrame, language_code: str) -> list[
    AllStyloMetrixFeaturesPL | AllStyloMetrixFeaturesEN]:
    model_instances = []
    for index, row in metrics_df.iterrows():
        if language_code == "pl":
            model_instance = AllStyloMetrixFeaturesPL(with_prepare=True, **row.to_dict())
        elif language_code == "en":
            model_instance = AllStyloMetrixFeaturesEN(with_prepare=True, **row.to_dict())
        else:
            raise ValueError(f"Language {language_code} is not supported")

        model_instances.append(model_instance)

    return model_instances


def calculate_perplexity_old(text: str, language_word_probabilities: Dict[str, float]) -> float:
    """
    Calculate perplexity for a given text based on a language model.

    Parameters:
    - text (str): The input text.
    - language_model (Dict[str, float]): A language model that provides probabilities for each word.

    Returns:
    - float: Perplexity value.
    """
    words = text.split()
    N = len(words)
    log_prob_sum = 0.0

    for word in words:
        # Assuming language_model is a dictionary with word probabilities
        # If you have a language model from a library, adjust this part accordingly
        word_prob = language_word_probabilities.get(word, 1e-10)  # Use a small default probability for unseen words
        log_prob_sum += math.log2(word_prob)

    average_log_prob = log_prob_sum / N
    perplexity = 2 ** (-average_log_prob)

    return perplexity


def calculate_perplexity(text: str, language_code: str, per_token: Optional[str] = "word",
                         return_base_ppl: bool = False, return_both: bool = False,
                         force_use_cpu: bool = False) -> Union[Optional[float], Tuple[float, float]]:
    text = replace_links_with_text(text)
    if per_token not in ["word", "char"]:
        raise ValueError("per_token must be either 'word' or 'char'")

    match per_token:
        case "word":
            denominator = len(text.split())
        case "char":
            denominator = len(text)
        case _:
            raise ValueError("per_token must be either 'word' or 'char'")

    if language_code == "pl":
        from config import PERPLEXITY_POLISH_TOKENIZER, PERPLEXITY_POLISH_MODEL
        tokenizer = PERPLEXITY_POLISH_TOKENIZER
        model = PERPLEXITY_POLISH_MODEL
    elif language_code == "en":
        from config import PERPLEXITY_ENGLISH_TOKENIZER, PERPLEXITY_ENGLISH_MODEL
        tokenizer = PERPLEXITY_ENGLISH_TOKENIZER
        model = PERPLEXITY_ENGLISH_MODEL
    else:
        raise ValueError("Language code must be either 'pl' or 'en'")

    # Determine device based on force_use_cpu flag and GPU availability
    if force_use_cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.eval()

    max_length = model.config.n_positions
    encodings = recursively_calculate_encodings(text, tokenizer, max_length) # tokenizer(text, return_tensors="pt")

    stride = 512
    if isinstance(encodings, dict):
        seq_len = encodings['input_ids'].size(1)
    else:
        seq_len = encodings.input_ids.size(1)

    sum_nll = 0.0
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        if isinstance(encodings, dict):
            input_ids = encodings['input_ids'][:, begin_loc:end_loc].to(device)
        else:
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss.detach().cpu().item()

        sum_nll += neg_log_likelihood

        # Clean up
        del outputs, input_ids, target_ids
        if device == 'cuda':
            torch.cuda.empty_cache()

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if prev_end_loc == 0:
        return None

    sum_nll_tensor = torch.tensor(sum_nll)

    if return_base_ppl:
        return float(sum_nll_tensor)

    ppl = float(torch.exp(sum_nll_tensor / denominator).float())
    if return_both:
        return float(sum_nll_tensor), ppl

    return ppl


import torch
import warnings


import torch
import warnings

def recursively_calculate_encodings(text: str, tokenizer, max_length):
    encodings = tokenizer(text, return_tensors="pt")

    if len(encodings.encodings[0].ids) > max_length:
        # Split the text into three parts:
        # first half, middle (1/4 to 3/4), second half
        length = len(text)
        half = length // 2
        quarter = length // 4
        three_quarters = 3 * length // 4

        text1 = text[:half]
        text2 = text[quarter:three_quarters]
        text3 = text[half:]

        # Recursively process each chunk
        encodings1 = recursively_calculate_encodings(text1, tokenizer, max_length)
        encodings2 = recursively_calculate_encodings(text2, tokenizer, max_length)
        encodings3 = recursively_calculate_encodings(text3, tokenizer, max_length)

        # Now slice each chunk's encodings as specified:
        # first: [:3/4]
        # second: [1/4:3/4]
        # third: [1/4:]
        combined = {}
        for key in encodings1.keys():
            L1 = encodings1[key].size(1)
            L2 = encodings2[key].size(1)
            L3 = encodings3[key].size(1)

            part1 = encodings1[key][:, :int(0.75 * L1)]
            part2 = encodings2[key][:, int(0.25 * L2):int(0.75 * L2)]
            part3 = encodings3[key][:, int(0.25 * L3):]

            combined[key] = torch.cat([part1, part2, part3], dim=1)

        encodings = combined

    return encodings


# https://github.com/AIAnytime/GPT-Shield-AI-Plagiarism-Detector

def calculate_burstiness(lemmatize_text: str, language_code: str) -> float:
    tokens = remove_stopwords_punctuation_emojis_and_splittings(lemmatize_text, language_code)

    word_freq = nltk.FreqDist(tokens)
    avg_freq = float(sum(word_freq.values()) / len(word_freq))
    variance = float(sum((freq - avg_freq) ** 2 for freq in word_freq.values()) / len(word_freq))

    burstiness_score = variance / (avg_freq ** 2)
    return burstiness_score

# https://arxiv.org/pdf/2310.05030
def calculate_burstiness_as_in_papers(lemmatized_text: str, language_code: str) -> float:
    # Tokenize the text
    tokens = remove_stopwords_punctuation_emojis_and_splittings(lemmatized_text, language_code)

    # Create a dictionary to store the positions of each word
    word_positions = defaultdict(list)

    for index, word in enumerate(tokens):
        word_positions[word].append(index)

    inter_arrival_times = []

    for positions in word_positions.values():
        if len(positions) > 1:
            inter_arrival_times.extend(np.diff(positions))

    if not inter_arrival_times:
        return 0.0

    mean_iat = np.mean(inter_arrival_times)
    std_iat = np.std(inter_arrival_times)

    burstiness = (std_iat - mean_iat) / (std_iat + mean_iat)

    return burstiness


# https://github.com/thinkst/zippy/blob/main/burstiness.py
def calc_distribution_sentence_length(sentences: List[str]) -> Tuple[
    Tuple[float, float, float], Tuple[float, float, float]]:
    '''
    Given a list of sentences returns the standard deviation, variance and average of sentence length in terms of both chars and words
    '''
    lens = []
    for sentence in sentences:
        chars = len(sentence)
        if chars < 1:
            continue
        words = len(sentence.split(' '))
        lens.append((chars, words))
    char_data = [x[0] for x in lens]
    word_data = [x[1] for x in lens]
    char_length_distribution: Tuple[float, float, float] = (std(char_data), var(char_data), mean(char_data))
    word_length_distribution: Tuple[float, float, float] = (std(word_data), var(word_data), mean(word_data))
    return char_length_distribution, word_length_distribution


def calculate_burstiness_old(text: str, window_size: int, language_word_probabilities: Dict[str, float]) -> float:
    """
    Calculate burstiness of the text based on the standard deviation of perplexity over windows.

    Parameters:
    - text (str): Input text.
    - window_size (int): Size of the window for calculating perplexity.
    - language_model (LanguageModel): Language model object.

    Returns:
    - float: Burstiness value.
    """
    perplexities = []

    # Split the text into windows
    windows = [text[i:i + window_size] for i in range(0, len(text), window_size)]

    # Calculate perplexity for each window
    for window in windows:
        perplexity = calculate_perplexity(window, language_word_probabilities)
        perplexities.append(perplexity)

    # Calculate standard deviation of perplexity
    burstiness = np.std(perplexities)

    return burstiness


def extract_strings_from_html(html_text: str, ignore_links: bool = True) -> str:
    """
    Extract text strings from HTML.

    Parameters:
    - html_text (str): HTML text.

    Returns:
    - str with concatenated text strings.
    """
    html2text_handler.ignore_links = ignore_links
    return html2text_handler.handle(html_text)


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Parameters:
    - text (str): Input text.

    Returns:
    - str: Detected language or 'unknown'.
    """
    # lang, _ = langid.classify(text.lower())
    DetectorFactory.seed = 0
    try:
        lang = langdetect_detect(text)
    except LangDetectException:
        lang = 'unknown'
    return lang


def detect_language_by_voting(text):
    # Ensure consistent results with langdetect
    DetectorFactory.seed = 0

    # Initialize a list to store detected languages
    detected_languages = []

    # langdetect
    try:
        detected_languages.append(langdetect_detect(text))
    except LangDetectException:
        pass

    # langid
    try:
        langid_result = langid.classify(text)
        detected_languages.append(langid_result[0])
    except Exception:
        pass

    # cld2
    try:
        isReliable, _, details = cld2.detect(text)
        if isReliable:
            detected_languages.append(details[0][1])
    except Exception:
        pass

    # cld3
    try:
        cld3_result = cld3.get_language(text)
        if cld3_result is not None:
            detected_languages.append(cld3_result.language)
    except Exception:
        pass

    if len(detected_languages) == 0:
        return 'unknown'

    # Perform voting
    language_counts = Counter(detected_languages)
    most_common_language, _ = language_counts.most_common(1)[0]

    return most_common_language


def spelling_and_grammar_check(text: str, lang_code: str) -> Tuple[Dict[str, int], int]:
    if lang_code == "pl":
        from config import LANGUAGE_TOOL_PL
        tool = LANGUAGE_TOOL_PL
        from config import LANGUAGE_TOOL_EN
        tool_en = LANGUAGE_TOOL_EN
    elif lang_code == "en":
        from config import LANGUAGE_TOOL_EN
        tool = LANGUAGE_TOOL_EN
    else:
        raise ValueError(f"Language {lang_code} is not supported")
    # Matches sequences of 4+ dots with optional spaces between them.
    # This is common in listings at the start of a report.
    text = re.sub(r'(\s)?\.(?:\s*\.){3,}', '', text)

    matches = tool.check(text)
    error_categories = {}

    # Categorize and count errors
    number_of_skipped_errors = 0
    for match in matches:
        category = match.category
        if category == 'TYPOS':
            error_text = match.matchedText
            if is_abbreviation(error_text):
                # ignore this error
                number_of_skipped_errors += 1
                continue
            if lang_code == 'pl': # check if maybe the matched text is in english
                matches_en = tool_en.check(error_text)
                if len(matches_en) == 0:
                    number_of_skipped_errors += 1
                    continue
            if len(match.replacements) == 0: # assume that this is a proper noun
                number_of_skipped_errors += 1
                continue
        if category in error_categories:
            error_categories[category] += 1
        else:
            error_categories[category] = 1

    return error_categories, len(matches) - number_of_skipped_errors


def measure_text_features(text: str) -> Dict[str, Optional[int]]:
    features = {
        'double_spaces': len(re.findall(r'\s{2,}', text)),
        'no_space_after_punctuation': len(re.findall(r'[,.:;!?](?=[^\s])', text)),
        'emojis': len(re.findall(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]',
            text)),
        'question_marks': len(re.findall(r'\?', text)),
        'exclamation_marks': len(re.findall(r'!', text)),
        'double_question_marks': len(re.findall(r'\?\?', text)),
        'double_exclamation_marks': len(re.findall(r'!!', text))
    }
    return features


def perform_full_analysis(text: str, lang_code: str) -> Union[AttributeNoDBParametersPL, AttributeNoDBParametersEN]:
    perplexity_base, perplexity = None, None #calculate_perplexity(text, lang_code, return_both=True)

    lem_text, _ = lemmatize_text(text, lang_code)
    lem_text = lem_text.strip()
    lemmatized_text = lem_text
    burstiness = calculate_burstiness(lem_text, lang_code)
    burstiness2 = calculate_burstiness_as_in_papers(lem_text, lang_code)


    words = [token for token in text.split() if token not in string.punctuation]
    number_of_words = len(words)
    char_data = [len(word) for word in words]
    number_of_characters = len(text)
    average_word_char_length = sum(char_data) / len(char_data)
    standard_deviation_word_char_length = std(char_data)
    variance_word_char_length = var(char_data)

    split_sentences: List[str] = split_into_sentences(text, lang_code)
    number_of_sentences = len(split_sentences)
    char_length_distribution, word_length_distribution = calc_distribution_sentence_length(split_sentences)
    average_sentence_word_length = word_length_distribution[2]
    standard_deviation_sentence_word_length = word_length_distribution[0]
    variance_sentence_word_length = word_length_distribution[1]
    average_sentence_char_length = char_length_distribution[2]
    standard_deviation_sentence_char_length = char_length_distribution[0]
    variance_sentence_char_length = char_length_distribution[1]

    punctuation = len([char for char in text if char in ".,!?;:"])
    punctuation_density = punctuation / number_of_characters
    punctuation_per_sentence = punctuation / number_of_sentences

    text_features = measure_text_features(text)
    double_spaces = text_features['double_spaces']
    no_space_after_punctuation = text_features['no_space_after_punctuation']
    emojis = text_features['emojis']
    question_marks = text_features['question_marks']
    exclamation_marks = text_features['exclamation_marks']
    double_question_marks = text_features['double_question_marks']
    double_exclamation_marks = text_features['double_exclamation_marks']
    text_errors_by_category, number_of_errors = spelling_and_grammar_check(text, lang_code)
    stylometrix_metrics = stylo_metrix_analysis([text], lang_code)[0]


    if lang_code == "en":
        pos_eng_tags = count_pos_tags_eng(text)
        sentiment_eng = sentiment_score_eng(text)
        return AttributeNoDBParametersEN(perplexity=perplexity, perplexity_base=perplexity_base,
                                         burstiness=burstiness, burstiness2=burstiness2,
                                         average_sentence_word_length=average_sentence_word_length,
                                         standard_deviation_sentence_word_length=standard_deviation_sentence_word_length,
                                         variance_sentence_word_length=variance_sentence_word_length,
                                         standard_deviation_sentence_char_length=standard_deviation_sentence_char_length,
                                         variance_sentence_char_length=variance_sentence_char_length,
                                         average_sentence_char_length=average_sentence_char_length,
                                         standard_deviation_word_char_length=standard_deviation_word_char_length,
                                         variance_word_char_length=variance_word_char_length,
                                         average_word_char_length=average_word_char_length,
                                         punctuation=punctuation, punctuation_per_sentence=punctuation_per_sentence,
                                         punctuation_density=punctuation_density, number_of_sentences=number_of_sentences,
                                         number_of_words=number_of_words, number_of_characters=number_of_characters,
                                         double_spaces=double_spaces, no_space_after_punctuation=no_space_after_punctuation,
                                         emojis=emojis, question_marks=question_marks, exclamation_marks=exclamation_marks,
                                         double_question_marks=double_question_marks,
                                         double_exclamation_marks=double_exclamation_marks,
                                         text_errors_by_category=text_errors_by_category, number_of_errors=number_of_errors,
                                         lemmatized_text=lemmatized_text, pos_eng_tags=pos_eng_tags, sentiment_eng=sentiment_eng,
                                         stylometrix_metrics=stylometrix_metrics.dict() if stylometrix_metrics is not None else None)
    elif lang_code == "pl":
        pos_eng_tags = None
        sentiment_eng = None
        return AttributeNoDBParametersPL(perplexity=perplexity, perplexity_base=perplexity_base,
                                         burstiness=burstiness, burstiness2=burstiness2,
                                         average_sentence_word_length=average_sentence_word_length,
                                         standard_deviation_sentence_word_length=standard_deviation_sentence_word_length,
                                         variance_sentence_word_length=variance_sentence_word_length,
                                         standard_deviation_sentence_char_length=standard_deviation_sentence_char_length,
                                         variance_sentence_char_length=variance_sentence_char_length,
                                         average_sentence_char_length=average_sentence_char_length,
                                         standard_deviation_word_char_length=standard_deviation_word_char_length,
                                         variance_word_char_length=variance_word_char_length,
                                         average_word_char_length=average_word_char_length,
                                         punctuation=punctuation, punctuation_per_sentence=punctuation_per_sentence,
                                         punctuation_density=punctuation_density, number_of_sentences=number_of_sentences,
                                         number_of_words=number_of_words, number_of_characters=number_of_characters,
                                         double_spaces=double_spaces, no_space_after_punctuation=no_space_after_punctuation,
                                         emojis=emojis, question_marks=question_marks, exclamation_marks=exclamation_marks,
                                         double_question_marks=double_question_marks,
                                         double_exclamation_marks=double_exclamation_marks,
                                         text_errors_by_category=text_errors_by_category, number_of_errors=number_of_errors,
                                         lemmatized_text=lemmatized_text, pos_eng_tags=pos_eng_tags, sentiment_eng=sentiment_eng,
                                         stylometrix_metrics=stylometrix_metrics.dict() if stylometrix_metrics is not None else None)
    else:
        raise ValueError(f"Language {lang_code} is not supported")

