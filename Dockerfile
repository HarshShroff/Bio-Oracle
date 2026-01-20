# Use an official Python runtime as a parent image
# Utilizing slim version for smaller footprint, ensuring python 3.11 for compatibility
FROM python:3.11-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
# gcc/python3-dev might be needed for some compiled extensions (like numpy/pandas/cv2) if wheels aren't found
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' bio-user
USER bio-user

# Entrypoint to allow running arguments directly
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
