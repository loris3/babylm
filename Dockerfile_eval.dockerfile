FROM nvidia/cuda:12.6.3-base-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3.10 \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*







ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages


RUN  pip install --no-cache-dir --upgrade pip setuptools wheel

RUN git clone https://github.com/babylm/evaluation-pipeline-2025 && \
    cd evaluation-pipeline-2025 && \
    sed -i 's/^ipython==[0-9.]\+/ipython/' requirements.txt


RUN cd evaluation-pipeline-2025 && \
    pip install --no-cache-dir -r requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /evaluation-pipeline-2025
COPY evaluation_data evaluation_data
RUN python3 -m pip install --no-cache-dir "huggingface_hub[cli]" hf_transfer
RUN huggingface-cli login --token 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



RUN python -c "import nltk; nltk.download('punkt_tab')"

RUN ls
RUN python -m evaluation_pipeline.ewok.dl_and_filter

RUN ./evaluation_pipeline/devbench/download_data.sh


CMD ["python3"]
 