# Use BuildKit for multi-stage builds

# syntax=docker/dockerfile:1.4

# Stage 1: Build stage
FROM python:3.11-slim AS build

# Set a working directory for the build stage
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install the required dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app

RUN mkdir -p /app/logs

RUN mkdir -p /app/models

RUN mkdir -p /app/results

# Create a non-root user
RUN useradd -m nonrootuser

# Change ownership of the directory to the non-root user
RUN chown -R nonrootuser:nonrootuser /app

# Switch to the non-root user
USER nonrootuser

# Expose the necessary ports
EXPOSE 5000 8000

# Run the application as the non-root user
CMD bash -c "mlflow server --host 0.0.0.0 --port 8000 & python -m mle_training_bhujith"
