FROM python:3.11-slim
WORKDIR /app
RUN adduser --disabled-password --gecos "" myuser
USER myuser
ENV PATH="/home/myuser/.local/bin:$PATH"
ENV GOOGLE_GENAI_USE_VERTEXAI=1
ENV GOOGLE_CLOUD_PROJECT=my-project-2026-hackthon
ENV GOOGLE_CLOUD_LOCATION=europe-west1
RUN pip install google-adk==1.28.0 google-auth opencc-python-reimplemented
COPY --chown=myuser:myuser "apac_expense_manager/" "/app/agents/apac_expense_manager/"
EXPOSE 8000
CMD adk web --port=8000 --host=0.0.0.0 --session_service_uri=memory:// --artifact_service_uri=memory:// "/app/agents"
