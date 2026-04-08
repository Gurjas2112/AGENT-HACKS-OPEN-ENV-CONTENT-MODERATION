FROM python:3.11-slim

WORKDIR /app
COPY envs/devpulse_code_review_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY envs/devpulse_code_review_env/ /app/envs/devpulse_code_review_env/

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "envs.devpulse_code_review_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]