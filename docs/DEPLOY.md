# Hosting the redesigned app for free

The app is one container running one process. It needs a host that runs a
Docker image or a Python web server. Streamlit Community Cloud cannot run
it: that service only starts a Streamlit script.

Nothing here has been deployed or tested on a host. Docker is not installed
on the development machine, so the image has not been built either. Treat
the first deployment as the test.

## What the host must provide

| Need | Why |
|---|---|
| Docker, or Python 3.12 plus Node 22 at build time | `deploy/studio/Dockerfile` builds the front end, then runs Python |
| About 512 MB of memory | The server measured 229 MB after a run and every tab on a 3-instrument, 16-year backtest. Larger universes and the Excel export need more headroom |
| One instance, one worker | Sessions and caches live in the process |
| An environment variable | `QBT_PASSWORD`, so the app is not open to anyone with the link |
| Outbound internet | Yahoo Finance and Bank of Canada downloads |

## Before choosing a host

Free tiers changed several times in 2026 and the comparison articles
disagree with each other. Check the provider's own pricing page on the day.
What the sources agreed on in November 2026:

| Host | Free tier as reported | Catch |
|---|---|---|
| Render | Docker web service, 512 MB, 750 hours a month | Sleeps after 15 minutes idle; waking takes about a minute. Sources disagree on whether a card is needed |
| Hugging Face Spaces (Docker) | 2 vCPU, 16 GB, sleeps after 48 hours idle | One source says Docker Spaces need a paid plan since mid-2026; others do not. Private Spaces may also be paid |
| Koyeb | Unclear | Reports that the free compute tier was withdrawn in 2026 |

Render is the most consistently reported free option that fits. Its
512 MB is enough for the measured footprint but not generous.

Whatever the host: a free instance that sleeps loses every session when it
does. The first visit after a sleep waits for the host to wake, then shows
the start screen. Saved presets and the Research ledger are written to the
container's disk and are lost on every redeploy and restart; the app says
so and offers an export (Presets > Export all presets, Ledger > Export).

## Deploying on Render

You do these steps; they need your account.

1. Push the branch to GitHub. This does not touch `main` or the live
   Streamlit app:
   ```bash
   git push -u origin ui/terminal-redesign
   ```
2. On render.com: **New > Web Service**, connect the repository.
3. Branch: `ui/terminal-redesign`. Runtime: **Docker**.
   Dockerfile path: `deploy/studio/Dockerfile`. Docker context: `.`
4. Instance type: **Free**.
5. Environment: add `QBT_PASSWORD` with a password of your choice.
6. Health check path: `/api/health`.
7. Create the service. The first build takes several minutes.
8. Open the URL. On an iPhone, open it in Safari and use **Add to Home
   Screen** for a full-screen icon.

Do not use the `render.yaml` at the repository root for this: it describes
the classic Streamlit app.

## Running the container yourself

```bash
docker build -f deploy/studio/Dockerfile -t qbt-studio .
```

```bash
docker run -p 8000:8000 -e QBT_PASSWORD=choose-one qbt-studio
```

## Without Docker

Any machine with Python 3.12 and Node 22:

```bash
pip install -r requirements.txt -r server/requirements.txt
```

```bash
cd web && npm ci && npm run build && cd ..
```

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

## If the first deployment fails

- **Build fails in the Node stage**: the lock file and Node version must
  match; the image uses Node 22 and `npm ci`.
- **Starts, then is killed**: memory. Try a smaller universe first; if it
  holds, the host's limit is the issue.
- **The page loads but every action says the server did not answer**: the
  host is running more than one instance or worker. Set both to one.
- **"app.py no longer matches server/classic.py"** in the logs: `app.py` was
  changed. See `docs/HANDOFF.md`, section 7.

## Sources consulted

- [Free cloud hosting 2026, 14 platforms compared](https://snapdeploy.dev/blog/free-cloud-deployment-platforms-2026-comparison)
- [9 free Docker hosting platforms in 2026](https://flywp.com/blog/9769/best-free-docker-hosting-platforms/)
- [Free Docker hosting in 2026, 6 options compared](https://miget.com/blog/best-free-docker-hosting)
- [Hugging Face Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview)
- [Streamlit Community Cloud: manage your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
