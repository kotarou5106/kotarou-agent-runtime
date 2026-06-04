# app

Startup and user-facing application entrypoints.

- `main.py` is the Personal Agent OS application entrypoint.
- `startup/` contains startup-oriented assembly helpers.
- `channels/` and `web_api/` are reserved for future direct app-level adapters.

Application startup is wired through `main.py`, `app/main.py`, and `bootstrap/`.
