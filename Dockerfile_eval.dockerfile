FROM loris3/babylm  
WORKDIR /app
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/babylm/evaluation-pipeline-2025
RUN cd evaluation-pipeline-2025 && pip install --no-cache-dir -r requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app/evaluation-pipeline-2025
COPY evaluation_data evaluation_data
RUN huggingface-cli login --token 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

RUN python -c "import nltk; nltk.download('punkt_tab')"
RUN python -m evaluation_pipeline.ewok.dl_and_filter
CMD ["python3"]
