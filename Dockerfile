FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Make sure templates and static directories exist
RUN mkdir -p /app/templates /app/static

# Copy the rest of the application
COPY . .

# Create data directory for caching with proper permissions
RUN mkdir -p /app/data && \
    chmod -R 777 /app/data

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Run the application (no debug mode in container)
CMD ["python", "main.py", "--port=5000"] 