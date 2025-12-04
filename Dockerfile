# Use a base image with Python 3.10
FROM python:3.10-slim

# System dependencies (for pycld3 and spacy)
RUN apt-get update && \
    apt-get install -y libprotobuf-dev protobuf-compiler gcc g++ curl default-jre && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install SpaCy models required for StyloMetrix
RUN pip install -U pip setuptools wheel && \
    pip install -U spacy && \
    python -m spacy download en_core_web_trf && \
    python -m spacy download pl_core_news_lg

# Download and install pl_nask model
RUN curl -L -o /tmp/pl_nask.tar.gz "http://mozart.ipipan.waw.pl/~rtuora/spacy/pl_nask-0.0.7.tar.gz" && \
    pip install /tmp/pl_nask.tar.gz && \
    rm /tmp/pl_nask.tar.gz

# Install stylometrix (only after all dependencies are satisfied)
RUN pip install stylo_metrix

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Add project files to the container
COPY . .

# Create a histograms directory inside the container
RUN mkdir -p /app/api/histograms

# Default command
ENTRYPOINT ["bash", "api/run_with_uvicorn.sh"]
