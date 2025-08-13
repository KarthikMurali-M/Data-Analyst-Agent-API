# Dockerfile

# Start from a standard Python 3.11 base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install System-Level Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    git \
    default-jre \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Install 'uv', the fast Python package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install Python Dependencies using 'uv'
COPY requirements.txt .
# Use the full path and add the --system flag
RUN /root/.local/bin/uv pip install --no-cache-dir -r requirements.txt --system

# 4. Install Playwright Browsers
RUN playwright install

# 5. Copy Your Application Code
COPY . .

# 6. Expose the Port
EXPOSE 8000

# 7. Define the Startup Command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
