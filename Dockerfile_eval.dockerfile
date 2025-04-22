FROM loris3/babylm  
WORKDIR /app
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/babylm/evaluation-pipeline-2024
RUN cd evaluation-pipeline-2024 && pip install -e .
RUN pip install --no-cache-dir minicons
RUN pip install --no-cache-dir --upgrade accelerate
RUN pip install tiktoken



# Set default command
CMD ["python3"]
