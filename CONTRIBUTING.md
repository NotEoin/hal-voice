# Contributing

## Set up and test

```bash
uv run --frozen --extra test pytest -q # run the hermetic tests (no GPU, no network, no audio)
ruff check . && ruff format --check .  # lint and format
bash scripts/ascii-check.sh            # tracked text must be ASCII
```

CI runs the same four commands. A test that needs a GPU, a microphone or the network carries a
marker and is skipped unless you ask for it.

## Rules

- One logical change per commit, with its tests and doc updates. Every commit passes the
  commands above.
- Commit subjects start with a capital verb in the imperative: `Add ...`, `Fix ...`. No body.
- Every module opens with a docstring saying what it is for; public functions have a
  one-line docstring and type hints.
- Comments say why, not what. Every `noqa` names its rule and gives a reason.
- Every subprocess call and HTTP request has a timeout (`tests/test_timeouts.py` checks).
- ASCII only in code, docs and commit messages. Write non-ASCII text as escapes.
- A bug fix comes with a test that fails without it.
- Fixtures are synthetic: no recordings of real people, no film clips, no secrets.

## Reporting a security problem

See [SECURITY.md](SECURITY.md).
