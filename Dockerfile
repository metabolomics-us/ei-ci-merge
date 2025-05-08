# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY merge.py .

# Set the entrypoint to the Python script
ENTRYPOINT ["python", "merge.py"]