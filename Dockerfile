FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .

RUN  pip install --no-cache-dir --upgrade pip setuptools wheel
RUN  pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN  pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-build-isolation traker[fast]
RUN python3 -m spacy download en_core_web_sm



CMD ["python3"]



