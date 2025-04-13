# Base image
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    ninja-build \
    libopenblas-dev \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy the rest of the app
COPY . .

# Run the app
CMD ["streamlit", "run", "app_local.py", "--server.port=8501", "--server.address=0.0.0.0"]
