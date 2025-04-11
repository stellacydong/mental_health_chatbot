# Dockerfile: Streamlit + XGBoost + llama-cpp-python (Mistral-7B)

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy app code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Launch the Streamlit app
CMD ["streamlit", "run", "app_local.py", "--server.port=8501", "--server.address=0.0.0.0"]