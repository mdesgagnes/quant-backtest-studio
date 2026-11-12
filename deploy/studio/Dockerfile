# The redesigned app: one container, one process.
#
# Build from the repository root:
#   docker build -f deploy/studio/Dockerfile -t qbt-studio .
#   docker run -p 8000:8000 -e QBT_PASSWORD=choose-one qbt-studio
#
# Stage 1 builds the front end; stage 2 is Python only and serves it.

FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
# This path makes the app say that saved presets do not survive a redeploy
# (qbt/store.py checks for it), which is true of every free host.
WORKDIR /mount/src/app

COPY requirements.txt ./requirements.txt
COPY server/requirements.txt ./server-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r server-requirements.txt

COPY app.py ./
COPY qbt ./qbt
COPY server ./server
COPY configs ./configs
COPY --from=web /web/dist ./web/dist

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://localhost:%s/api/health' % os.environ.get('PORT','8000'))"

# One worker on purpose: sessions and the run cache live in this process.
CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
