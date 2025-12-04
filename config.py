import os

import nltk
import spacy

from analysis.custom_stylometrix import CustomStyloMetrix
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM

import language_tool_python

load_dotenv()

def init_nltk():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('pl196x')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

MINIMAL_SENTENCE_LENGTH = int(os.getenv("MINIMAL_SENTENCE_LENGTH", 3))
SUSPICIOUS_SENTENCE_LENGTH = int(os.getenv("SUSPICIOUS_SENTENCE_LENGTH", 25))
MAXIMAL_SENTENCE_LENGTH = int(os.getenv("MAXIMAL_SENTENCE_LENGTH", 150))

SPACY_POLISH_NLP_MODEL = None
SPACY_ENGLISH_NLP_MODEL = None

LANGUAGE_TOOL_PL = None
LANGUAGE_TOOL_EN = None

def init_language_tool_pl() -> None:
    global LANGUAGE_TOOL_PL
    LANGUAGE_TOOL_PL = language_tool_python.LanguageTool('pl-PL')


def init_language_tool_en() -> None:
    global LANGUAGE_TOOL_EN
    LANGUAGE_TOOL_EN = language_tool_python.LanguageTool('en-US')

def init_spacy_polish_nlp_model() -> None:
    global SPACY_POLISH_NLP_MODEL
    SPACY_POLISH_NLP_MODEL = spacy.load("pl_core_news_lg")

def init_spacy_english_nlp_model() -> None:
    global SPACY_ENGLISH_NLP_MODEL
    SPACY_ENGLISH_NLP_MODEL = spacy.load("en_core_web_trf")

SPACY_POLISH_NLP_MODEL_SMALL = None
SPACY_ENGLISH_NLP_MODEL_SMALL = None


def init_spacy_polish_nlp_model_small() -> None:
    global SPACY_POLISH_NLP_MODEL_SMALL
    SPACY_POLISH_NLP_MODEL_SMALL = spacy.load("pl_core_news_sm")

def init_spacy_english_nlp_model_small() -> None:
    global SPACY_ENGLISH_NLP_MODEL_SMALL
    SPACY_ENGLISH_NLP_MODEL_SMALL = spacy.load("en_core_web_sm")


PERPLEXITY_POLISH_MODEL = None
PERPLEXITY_ENGLISH_MODEL = None
PERPLEXITY_POLISH_GPT2_MODEL_ID = os.getenv("PERPLEXITY_POLISH_GPT2_MODEL_ID")
PERPLEXITY_POLISH_QRA_MODEL_ID = os.getenv("PERPLEXITY_POLISH_QRA_MODEL_ID")
PERPLEXITY_ENGLISH_GPT2_MODEL_ID = os.getenv("PERPLEXITY_ENGLISH_GPT2_MODEL_ID")
PERPLEXITY_MODEL_ID = os.getenv("PERPLEXITY_MODEL_ID")
PERPLEXITY_POLISH_TOKENIZER = None
PERPLEXITY_ENGLISH_TOKENIZER = None

def init_polish_perplexity_model(model_name: str = PERPLEXITY_POLISH_GPT2_MODEL_ID) -> None:
    global PERPLEXITY_POLISH_MODEL
    PERPLEXITY_POLISH_MODEL = AutoModelForCausalLM.from_pretrained(model_name)
    global PERPLEXITY_POLISH_TOKENIZER
    PERPLEXITY_POLISH_TOKENIZER = AutoTokenizer.from_pretrained(model_name)

def init_english_perplexity_model(model_name: str = PERPLEXITY_ENGLISH_GPT2_MODEL_ID) -> None:
    global PERPLEXITY_ENGLISH_MODEL
    PERPLEXITY_ENGLISH_MODEL = GPT2LMHeadModel.from_pretrained(model_name)
    global PERPLEXITY_ENGLISH_TOKENIZER
    PERPLEXITY_ENGLISH_TOKENIZER = GPT2TokenizerFast.from_pretrained(model_name)

RELATIVE_PATH_TO_PROJECT = os.getenv("RELATIVE_PATH_TO_PROJECT")

DICT_FILE_1GRAMS_PATH = os.getenv("DICT_FILE_1GRAMS_PATH")
DICT_FILE_ODM_PATH = os.getenv("DICT_FILE_ODM_PATH")
WORD_SET = None
def load_dictionaries():
    global WORD_SET
    WORD_SET = set()
    with open(DICT_FILE_1GRAMS_PATH, "r", encoding="utf-8") as f1:
        for line in f1:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                if parts[1]:
                    WORD_SET.add(parts[1])
            else:
                # If there's any malformed line, skip it
                continue

    # Load second file (comma-separated words per line)
    with open(DICT_FILE_ODM_PATH, "r", encoding="utf-8") as f2:
        for line in f2:
            # Split by comma and strip each word
            parts = [w.strip() for w in line.split(",")]
            for word in parts:
                if word:
                    WORD_SET.add(word)

STYLOMETRIX_PL_MODEL: CustomStyloMetrix | None = None
STYLOMETRIX_EN_MODEL: CustomStyloMetrix | None = None

def init_polish_stylometrix_model() -> None:
    global STYLOMETRIX_PL_MODEL
    STYLOMETRIX_PL_MODEL = CustomStyloMetrix('pl')

def init_english_stylometrix_model() -> None:
    global STYLOMETRIX_EN_MODEL
    STYLOMETRIX_EN_MODEL = CustomStyloMetrix('en')

def init_all_polish_models() -> None:
    init_polish_perplexity_model()
    init_nltk()
    init_spacy_polish_nlp_model()
    init_language_tool_pl()
    init_language_tool_en()
    init_polish_stylometrix_model()
    load_dictionaries()

def init_all_english_models() -> None:
    init_english_perplexity_model()
    init_nltk()
    init_spacy_english_nlp_model()
    init_language_tool_en()
    init_english_stylometrix_model()
    load_dictionaries()