# Anti GPT checker
This project aims to provide a reliable way of distinguishing between AI generated text and human written one. It is the implementation part of my Masters's Thesis at the Warsaw University of Technology. It is also a part of ANANAS project at the Warsaw University of Technology.

# Installation
- Python 3.10 or higher is required, with python-dev and pip installed.
- Start by installing stylometrix following the instructions in the [stylometrix repository](https://github.com/ZILiAT-NASK/StyloMetrix). This includes:
  - Installing `spacy` according to [their instructions](https://spacy.io/usage), select Polish and (optionally) English models, and pipeline for `accuracy`
  - Downloading and installing [pl_nask](http://mozart.ipipan.waw.pl/~rtuora/spacy/pl_nask-0.0.7.tar.gz) model `python -m pip install <PATH_TO_MODEL/pl_nask-0.0.7.tar.gz> `
  - Installing `stylometrix` `pip install stylometrix`
- Install the rest of the dependencies `pip install -r requirements.txt`
```bash