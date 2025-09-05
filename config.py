import os

import nltk
import spacy

import stylo_metrix as sm
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
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
"""
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_PORT = int(os.getenv("MONGODB_PORT",27017))
MONGODB_DB_NAME= os.getenv("MONGODB_DB_NAME")
MONGODB_EMAILS_SRC_COLLECTIONS = {
    "email_spam_data": os.getenv("MONGODB_COLLECTION_EMAIL_SPAM_DATASET"),
    "email_spam_assassin": os.getenv("MONGODB_COLLECTION_EMAIL_SPAM_ASSASSIN_DATASET"),
    "email_class_git": os.getenv("MONGODB_COLLECTION_EMAIL_CLASSIFICATION_GITHUB"),
}

MONGODB_AUTH_USER = os.getenv("MONGODB_AUTH_USER")
MONGODB_AUTH_PASS = os.getenv("MONGODB_AUTH_PASS")
MONGODB_AUTH_DB = os.getenv("MONGODB_AUTH_DB")
"""
"""
if MONGODB_AUTH_USER and MONGODB_AUTH_PASS and MONGODB_AUTH_DB:
    MONGO_CLIENT = MongoClient(host=MONGODB_URI, port=MONGODB_PORT, username=MONGODB_AUTH_USER, password=MONGODB_AUTH_PASS, authSource=MONGODB_AUTH_DB)
    ASYNC_MONGO_CLIENT = AsyncIOMotorClient(host=MONGODB_URI, port=MONGODB_PORT, username=MONGODB_AUTH_USER, password=MONGODB_AUTH_PASS, authSource=MONGODB_AUTH_DB)
else:
    MONGO_CLIENT = MongoClient(MONGODB_URI, MONGODB_PORT)
    ASYNC_MONGO_CLIENT = AsyncIOMotorClient(MONGODB_URI, MONGODB_PORT)
"""
ATTRIBUTES_COLLECTION_NAME = os.getenv("ATTRIBUTES_COLLECTION_NAME", "attributes")
LAB_REPORTS_COLLECTION_NAME = os.getenv("LAB_REPORTS_COLLECTION_NAME", "lab_reports")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_EMAIL_GENERATION_PROMPT_NAME = os.getenv("DEFAULT_EMAIL_GENERATION_PROMPT_NAME")
OPENAI_GPT3_5_MODEL_NAME = os.getenv("OPENAI_GPT3_5_MODEL_NAME")
OPENAI_GPT4_MODEL_NAME = os.getenv("OPENAI_GPT4_MODEL_NAME")
MAX_API_RETRIES = 3

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

STYLOMETRIX_PL_MODEL = None
STYLOMETRIX_EN_MODEL = None

def init_polish_stylometrix_model() -> None:
    global STYLOMETRIX_PL_MODEL
    STYLOMETRIX_PL_MODEL = sm.StyloMetrix('pl')

def init_english_stylometrix_model() -> None:
    global STYLOMETRIX_EN_MODEL
    STYLOMETRIX_EN_MODEL = sm.StyloMetrix('en')

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