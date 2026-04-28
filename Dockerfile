FROM embedding-server-base

WORKDIR /app

COPY app/ app/

EXPOSE 8000

CMD uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}
