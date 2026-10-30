# Run the voice server in a container or on the host

`hal-tts.service` runs `hal-voice tts-server`. By default (`tts.server.isolation: container`) the
server runs in a GPU container with no network, read-only mounts and no capabilities. Host mode
is for machines where that is not possible, and it checks the model files first.

## Use the container (default)

1. Install docker and the NVIDIA Container Toolkit, then check the runtime.

   ```bash
   docker info --format '{{json .Runtimes}}'   # must list "nvidia"
   ```

2. Build the image once per hal-voice version.

   ```bash
   hal-voice tts-image build    # several GB; needs the network
   ```

3. Start the server.

   ```bash
   systemctl --user restart hal-tts.service   # or: hal-voice tts-server
   ```

The API stays on `127.0.0.1:9880` through a small forwarder, so `hal-say` works unchanged.

## Use host mode

1. Set it in your own config file, never in the packaged defaults.

   ```yaml
   # ~/.config/hal-voice/config.yaml
   tts:
     server:
       isolation: host
   ```

2. Review the checkpoints, then pin them. This is a human step.

   ```bash
   hal-voice tts-pins record    # writes ~/.config/hal-voice/models.lock
   ```

3. Start the server. It verifies the pins first and exits 1 naming any file that changed.

   ```bash
   hal-voice tts-pins verify    # also runs on every tts-server start
   ```

Host mode keeps torch's `weights_only=True` loading. Unverified: whether every GPT-SoVITS
checkpoint loads that way; check by starting the server once after `record`.

## If something goes wrong

| Symptom | Fix |
| --- | --- |
| `docker is not installed` | Install docker, or switch to host mode |
| `docker has no nvidia runtime` | Run `nvidia-ctk runtime configure --runtime=docker` and restart docker |
| `XDG_RUNTIME_DIR is not set` | Start the server from a login session or user unit |
| `refusing to start the host TTS server: ... does not match its pin` | Find out why the file changed; if it is yours, run `tts-pins record` again |
| `models.lock not found` | Run `hal-voice tts-pins record` |
| Host mode stops while loading a checkpoint | That file needs pickle loading; use container mode |
