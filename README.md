# Anti GPT checker
This project aims to provide a reliable way of distinguishing between AI generated text and human written one. It is the implementation part of my Masters's Thesis at the Warsaw University of Technology. It is also a part of ANANAS project at the Warsaw University of Technology.

# Installation
- Python 3.10 or higher is required, with python-dev and pip installed.
- Start by installing stylometrix following the instructions in the [stylometrix repository](https://github.com/ZILiAT-NASK/StyloMetrix). This includes:
  - Installing `spacy` according to [their instructions](https://spacy.io/usage), select Polish and (optionally) English models, and pipeline for `accuracy`
  - Downloading and installing [pl_nask](http://mozart.ipipan.waw.pl/~rtuora/spacy/pl_nask-0.0.7.tar.gz) model `python -m pip install <PATH_TO_MODEL/pl_nask-0.0.7.tar.gz> `
  - Installing `stylometrix` `pip install stylometrix`
- Initialize Polish dictionaries:
  - [SJP](https://sjp.pl/sl/odmiany/) download the newest available
  - [NKJP (1-gram)](https://zil.ipipan.waw.pl/NKJPNGrams?action=AttachFile&do=get&target=1grams.gz)
  - Unpack the archives and move files `odm.txt` and `1grams` into `data` directory inside the project
- Install the rest of the dependencies `pip install -r requirements.txt`
  - If install fails on `pycld3` run `sudo apt-get install -y libprotobuf-dev protobuf-compiler`


# .env file
Create a .env file in the root directory of the project with the following fields:
```bash
MONGODB_URI=""
MONGODB_PORT=""
MONGODB_DB_NAME=""
MONGODB_AUTH_USER=""
MONGODB_AUTH_PASS=""
MONGODB_AUTH_DB="" # usually admin
PERPLEXITY_POLISH_GPT2_MODEL_ID="sdadas/polish-gpt2-xl"
PERPLEXITY_POLISH_QRA_MODEL_ID="OPI-PG/Qra-13b"
PERPLEXITY_ENGLISH_GPT2_MODEL_ID="openai-community/gpt2-large"
RELATIVE_PATH_TO_PROJECT="" # relative path to the project directory from working directory
DICT_FILE_1GRAMS_PATH="data/1grams"
DICT_FILE_ODM_PATH="data/odm.txt"
```