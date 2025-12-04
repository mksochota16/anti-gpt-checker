#!/bin/bash
set -e  # Exit on error

# Update system and install dependencies
sudo apt-get update && \
sudo apt-get install -y libprotobuf-dev protobuf-compiler gcc g++ curl default-jre && \
sudo rm -rf /var/lib/apt/lists/*

# Go to app directory (create if missing)
mkdir -p ~/app
cd ~/app

# Set up venv
python3.10 -m venv anti-gpt-venv && source anti-gpt-venv/bin/activate

# Upgrade pip, setuptools, wheel
pip install -U pip setuptools wheel

# Install SpaCy and models
pip install -U spacy
python -m spacy download en_core_web_trf
python -m spacy download pl_core_news_lg

# Download and install pl_nask model
curl -L -o /tmp/pl_nask.tar.gz "http://mozart.ipipan.waw.pl/~rtuora/spacy/pl_nask-0.0.7.tar.gz"
pip install /tmp/pl_nask.tar.gz
rm /tmp/pl_nask.tar.gz

# Install stylometrix
pip install stylo_metrix

pip install -r requirements.txt

echo "Environment setup complete."
