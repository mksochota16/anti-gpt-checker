import os

import nltk
import spacy
from dotenv import load_dotenv
from pymongo import MongoClient
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM

import language_tool_python

load_dotenv()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('pl196x')
nltk.download('wordnet')
nltk.download('punkt_tab')

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_PORT = int(os.getenv("MONGODB_PORT"))
MONGODB_DB_NAME= os.getenv("MONGODB_DB_NAME")

MONGODB_AUTH_USER = os.getenv("MONGODB_AUTH_USER")
MONGODB_AUTH_PASS = os.getenv("MONGODB_AUTH_PASS")
MONGODB_AUTH_DB = os.getenv("MONGODB_AUTH_DB")

if MONGODB_AUTH_USER and MONGODB_AUTH_PASS and MONGODB_AUTH_DB:
    MONGO_CLIENT = MongoClient(host=MONGODB_URI, port=MONGODB_PORT, username=MONGODB_AUTH_USER, password=MONGODB_AUTH_PASS, authSource=MONGODB_AUTH_DB)
else:
    MONGO_CLIENT = MongoClient(MONGODB_URI, MONGODB_PORT)
ATTRIBUTES_COLLECTION_NAME = os.getenv("ATTRIBUTES_COLLECTION_NAME", "attributes")
LAB_REPORTS_COLLECTION_NAME = os.getenv("LAB_REPORTS_COLLECTION_NAME", "lab_reports")

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

RELATIVE_PATH_TO_PROJECT = os.getenv("RELATIVE_PATH_TO_PROJECT") # needed for notebooks to work properly




