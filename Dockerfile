FROM python:3.10-slim

WORKDIR /app

# Ensure Python output is UTF-8 encoded
ENV PYTHONIOENCODING=utf-8

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY baseline/ ./baseline/

# Build the index during the image build process
RUN python -m src.app index

CMD ["python", "-m", "src.app", "evaluate"]
