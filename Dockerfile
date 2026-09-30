FROM python:3.11-slim

WORKDIR /app

RUN adduser --disabled-password --gecos "" myuser
USER myuser
ENV PATH="/home/myuser/.local/bin:$PATH"

# Vertex AI environment (overridden by Cloud Run env vars)
ENV GOOGLE_GENAI_USE_VERTEXAI=1
ENV GOOGLE_CLOUD_PROJECT=your-gcp-project-id
ENV GOOGLE_CLOUD_LOCATION=europe-west1

# Install dependencies
RUN pip install google-adk==1.28.0 google-auth google-cloud-bigquery opencc-python-reimplemented

# Copy agent code
COPY --chown=myuser:myuser "apac_expense_manager/" "/app/agents/apac_expense_manager/"

EXPOSE 8000

# ADK web server — same CMD as adk deploy cloud_run generates
CMD adk web --port=8000 --host=0.0.0.0 --session_service_uri=memory:// --artifact_service_uri=memory:// "/app/agents"
