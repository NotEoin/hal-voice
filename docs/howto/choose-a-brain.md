# Choose a brain

hal-voice has no language model of its own. The brain is whatever answers your request after it is
transcribed. This page shows how to pick one, point it at a project, and check the choice.

## Steps

1. Pick the mode in `~/.config/hal-voice/config.yaml` (see [configuration](../user/configuration.md)):

   | `brain.mode` | Who answers | Tools |
   | --- | --- | --- |
   | `hal` (default) | Hal's boxed assistant agent, started as `hal acp --channel voice --toolset assistant` (`brain.hal_command`) | The tools of your installed Hal packs, through Hal's broker |
   | `openai` | Any OpenAI-compatible server at `brain.base_url`: llama-server, Ollama's `/v1`, vLLM or a hosted API | None |
   | `claude` | A headless Claude Code session, run for every turn | That project's tools and memory |

2. For `openai`, set the server URL. There is no default: with an empty `base_url`, hal-voice stops
   and names the key to set.

   ```yaml
   # ~/.config/hal-voice/config.yaml: a local llama-server, no Hal
   brain:
     mode: openai
     base_url: http://127.0.0.1:8080/v1
     api_key_file: ""   # a file holding the bearer token, if the server needs one
   ```

3. For `claude`, choose the directory Claude Code runs in. It reads that directory's `CLAUDE.md`,
   `.mcp.json` and `.claude/` settings. The default is `claude.project_dir`, else the vault named in
   `~/.config/hal/config.toml` (`[hal].vault`), else the current directory.

   ```bash
   ./hal-voice --project ~/code/my-project   # use that project's memory and tools
   ./hal-voice connection                    # show what it resolved, and why
   ```

4. Try a mode for one run without editing the file:

   ```bash
   HALVOICE_BRAIN_MODE=claude ./hal-voice                  # Claude Code for every turn
   HALVOICE_BRAIN_MODEL=mistral-small3.2:24b ./hal-voice   # another model, this run only
   ```

5. Check the setup.

   ```bash
   ./hal-voice doctor   # the "brain" and "Claude Code connection" lines show the result
   ```

## Rules each mode follows

- `hal`: when a tool call needs approval, hal-voice reads out the host's summary and accepts only a
  spoken "yes" or "allow" as the whole reply. Anything else denies, and so does
  `brain.voice_approvals: false`. If `hal` is missing or exits twice, hal-voice stops with an
  error. It never falls back to another brain. Checked by `tests/test_acp_brain.py`.
- `openai`: the last `brain.history_turns` exchanges are kept (8 when empty).
- `claude`: Claude Code runs on the host, outside Hal's sandbox, and hal-voice logs a warning at
  start. `claude.permission_mode` is `default`, so headless Claude Code refuses anything that needs
  a permission; set `acceptEdits` to let it change files. To follow a Hal install's profile, set
  `claude.hal_root` to a checkout with a `hal.conf` (or run `./setup-voice.sh --hal auto`). For a
  setup that is not Hal, set `claude.env_file` to a `KEY=VALUE` file of `ANTHROPIC_*` settings.
- `openai` and `claude`: a turn still running after `brain.turn_timeout_s` (300 s) ends with "That
  took too long". In `claude` mode its child processes are killed.

## Symptoms and fixes

| Symptom | Fix |
| --- | --- |
| hal-voice stops and names `brain.base_url` | Set `brain.base_url` in `openai` mode. |
| hal-voice stops because `hal` is missing | Install Hal and check that `hal` is on the `PATH`, or choose another `brain.mode`. |
| A tool call is denied | Answer with the single word "yes" or "allow" when the approval is read out. |
| Claude Code cannot edit files | Set `claude.permission_mode: acceptEdits`, knowing it then changes files without asking. |
| `hal-voice.service` cannot find `claude` | The unit sets no `PATH`; set `claude.command` to its absolute path. |
| A config key is rejected | The key was removed. Keys such as `brain.ollama_url` and `brain.num_ctx` are Hal's settings now. |
