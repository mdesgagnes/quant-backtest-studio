# Working rules for this repository

This repo is being moved from a Streamlit interface (`app.py`) to a new one
(`server/` + `web/`). The engine (`qbt/`) is correct and is not being
changed. Read `docs/HANDOFF.md` before starting work; it has the plan, the
patterns and the ticket list.

## Rules

1. **Parity is not negotiable.** The new app must show the same figures,
   tables, chart data, notes and exports as the Streamlit app.
   `docs/ui-parity-checklist.md` is the contract. If something cannot be
   reproduced, stop and ask the owner; do not approximate.
2. **The engine is out of scope.** Do not edit anything under `qbt/`, and do
   not edit `app.py`. If a screen needs data the engine does not expose, add
   an adapter under `server/` and explain it in the commit message.
3. **Work on the branch `ui/terminal-redesign` only.** Never push to `main`,
   never merge, never open a pull request unless the owner asks. `main`
   deploys the live app.
4. **Run the golden tests after every step**, and the whole suite before
   every commit:
   ```bash
   pytest tests/golden/test_golden_service.py tests/test_server_api.py -q   # seconds
   pytest -q                                                               # about 7 minutes
   ```
   A golden failure means the new code is wrong. Never re-record
   `tests/golden/expected/` to make a test pass.
5. **Follow `docs/DESIGN.md`.** Colours and sizes come from
   `web/src/styles/tokens.css` only. No new colour, font or chart library
   without the owner's agreement.
6. **The browser formats nothing.** Every figure is formatted in Python, in
   `server/views/`, with the expression `app.py` uses. TypeScript lays out
   and draws.
7. **Keep the classic app usable.** `streamlit run app.py` must keep working
   until the owner retires it.
8. **Open questions go to the owner.** The unresolved ones are U1 to U5 at
   the end of the parity checklist. Do not decide them by default.

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
| `qbt/` | Engine and exports. Do not edit |
| `app.py` | Classic Streamlit app. Do not edit; read it as the specification |
| `server/pipeline.py` | The run, lifted from `app.py` |
| `server/views/` | One file per screen: builds the screen's blocks |
| `server/serialize.py` | Blocks to JSON |
| `server/app.py` | HTTP routes |
| `web/src/` | Front end |
| `tests/golden/` | Reference runs and the harness |
| `docs/` | Checklist, design system, handoff, performance baseline |

## Style

- Python: match the surrounding code; keep `app.py`'s comments when lifting
  a block, so the two can be compared.
- TypeScript: strict; no `any`; components small enough to read in one
  screen.
- Text shown to users: sentence case, plain words, no em dashes, wording
  copied from `app.py` unless the checklist says otherwise.
- Commit messages: what changed and why, and which checklist items it
  covers.
