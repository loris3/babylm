FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# Install Python and required packages in one layer
RUN apt-get update && apt-get install -y \
    python3.10 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install dependencies in one step to reduce layers
COPY requirements.txt .

# Set up virtual environment and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-build-isolation traker[fast]

# Set environment variables for convenience


# Copy rest of the app if needed (you might want to add this if you have app code)
# COPY . .

CMD ["python"]
