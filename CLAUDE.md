# Working rules for this repository

The interface has been rebuilt: a new server (`server/`) and front end
(`web/`) replace the Streamlit page, over the same engine (`qbt/`). Read
`docs/HANDOFF.md` before starting work.

## Rules

1. **Parity is not negotiable.** The new app must show the same figures,
   tables, chart data, notes and exports as the classic app.
   `docs/ui-parity-checklist.md` is the contract. If something cannot be
   reproduced, stop and ask the owner; do not approximate.
2. **The engine is out of scope.** Do not edit anything under `qbt/`.
3. **`app.py` is not edited either.** It is still the classic Streamlit app,
   and it is also the screen logic the new server runs
   (`server/classic.py` reads it and adjusts it in memory). A change to it
   changes both apps and can stop the new one from starting. If screen
   logic must change, ask the owner first, then update the adjustments in
   `server/classic.py` and the golden references together.
4. **Work on the branch `ui/terminal-redesign` only.** Never push to `main`,
   never merge, never open a pull request unless the owner asks. `main`
   deploys the live classic app.
5. **Run the golden tests after every step**, and the whole suite before
   every commit:
   ```bash
   pytest tests/test_server_app.py tests/test_server_api.py tests/golden/test_golden_service.py -q
   pytest tests/golden/test_golden_app.py -q
   pytest -q
   ```
   A golden failure means the new code is wrong. Never re-record
   `tests/golden/expected/` to make a test pass.
6. **Follow `docs/DESIGN.md`.** Colours and sizes come from
   `web/src/styles/tokens.css` only. No new colour, font or chart library
   without the owner's agreement.
7. **The browser formats nothing.** Figures arrive as the text to show.
   TypeScript lays out and draws.
8. **Keep the classic app runnable** (`streamlit run app.py`) until the
   owner retires it.
9. **No secrets in the repository.** The password is the `QBT_PASSWORD`
   environment variable on the host.

## Commands

```bash
# Python: C:\Users\marcd\miniconda3\envs\quant\python.exe  (conda env "quant")
python -m uvicorn server.app:app --port 8000      # new app: API + built front end
cd web && npm run dev                             # front end with hot reload (proxies /api to :8000)
cd web && npm run build                           # type-check and build into web/dist
streamlit run app.py                              # classic app, port 8501
```

## Where things are

| Path | What |
|---|---|
| `qbt/` | Engine, exports, and three Streamlit view modules. Do not edit |
| `app.py` | Classic app and the screen logic of the new one. Do not edit |
| `server/shim.py` | Recording stand-in for the Streamlit API |
| `server/classic.py` | Runs `app.py` under the recorder; tab skipping and caching |
| `server/present.py` | Recorded nodes to blocks for the browser |
| `server/app.py` | HTTP routes and sessions |
| `server/views/results.py` | The dedicated Results screen |
| `server/pipeline.py`, `server/store.py` | Programmatic run API (not used by the interface) |
| `web/src/live.tsx` | Connection to the server |
| `web/src/blocks/` | One component per kind of block and control |
| `web/src/charts/` | Charts |
| `web/src/styles/` | Tokens and styles |
| `tests/golden/` | Reference runs, the Streamlit harness and the new-app driver |
| `docs/` | Checklist, design system, handoff, deployment, performance baseline |

## Style

- Python: match the surrounding code.
- TypeScript: strict; components small enough to read in one screen.
- Text shown to users: sentence case, plain words, no em dashes.
- Commit messages: what changed and why, and which checklist items it
  covers.
