# Run on Windows with WSL2

This page shows how to run hal-voice on a Windows 11 PC through WSL2, and which parts of
hal-voice work there. Native Windows and macOS are not supported: hal-voice needs Linux
audio (PortAudio over PulseAudio or PipeWire), systemd user units and UNIX sockets. WSL2 runs a
real Linux kernel and Windows 11 passes the microphone and speakers through WSLg, so it is the
supported route.

Support is new and has not been run on a Windows PC yet. The table after the steps marks every
feature as verified or unverified, and `hal-voice doctor` probes each one on your machine.

## Steps

1. Enable WSL and install Ubuntu. In an administrator PowerShell on Windows:

   ```powershell
   wsl --install -d Ubuntu-24.04   # installs WSL2 and the distro; reboot when asked
   wsl --update                    # WSLg (audio and windows) ships with the WSL package
   wsl --status                    # Default Version: 2
   ```

2. Keep hal-voice in the Linux filesystem, not under `/mnt/c`. Files on `/mnt/c` are slow to
   read and have no Unix permissions. Model downloads and the voice build read many files.

   ```bash
   cd ~ && pwd   # /home/<you>, inside the distro's own disk
   ```

3. Check the audio server. WSLg offers PulseAudio on a socket; Windows owns the devices.

   ```bash
   ls -l /mnt/wslg/PulseServer   # a socket; if missing, run `wsl --update` and `wsl --shutdown`
   sudo apt-get update && sudo apt-get install -y libportaudio2 pulseaudio-utils ffmpeg git
   pactl info | head -3          # Server String: /mnt/wslg/PulseServer
   ```

4. Install hal-voice inside Ubuntu:

   ```bash
   git clone https://github.com/NotEoin/hal-voice.git && cd hal-voice
   ./setup-voice.sh   # finds Python 3.11 or 3.12; Ubuntu 24.04 ships 3.12
   ```

5. Run the doctor. Under WSL2 it adds lines for the audio server, the display and systemd:

   ```bash
   hal-voice doctor    # every line says what works and what to do about what does not
   hal-voice devices   # lists a mic and an output device that WSLg provides
   ```

6. Optional: enable systemd, so the `hal-voice.service` and `hal-tts.service` units can run.
   Skip this if you start `hal-voice` by hand in a terminal.

   ```bash
   printf '[boot]\nsystemd=true\n' | sudo tee -a /etc/wsl.conf   # edit by hand if [boot] exists
   ```

   ```powershell
   wsl --shutdown   # run in PowerShell; stops every WSL distro, then reopen Ubuntu
   ```

7. Optional: GPU. Install the NVIDIA driver on Windows, not in Ubuntu. Inside Ubuntu,
   `nvidia-smi` then works. The default `gpu.mode: on_demand` runs on the CPU, so this is only
   needed for the `resident` mode. For sharing the card with other jobs, see the gpu-light
   how-to on WSL2.

   ```bash
   nvidia-smi   # lists your GPU; if not, update the Windows driver
   ```

8. Optional: the TTS container (`tts.server.isolation: container`). Use Docker Engine inside
   Ubuntu, or Docker Desktop with WSL integration for this distro. For a GPU container, install
   the NVIDIA Container Toolkit in Ubuntu.

   ```bash
   docker info --format '{{json .Runtimes}}'   # lists "nvidia" for a GPU container
   hal-voice tts-image build                   # builds the image
   ```

## What works under WSL2

"Unverified" means the design expects it to work, but nobody has run it on WSL2. Run the check
in the last column and report the result.

| Feature | Status | How to check |
| --- | --- | --- |
| Platform detection (`osrelease` says `microsoft` or `WSL`) | Verified with a fake `osrelease` in the test suite | `hal-voice doctor`: the `platform` line says WSL2 |
| Refusing to start on native Windows and macOS | Verified with a fake platform | Run `hal-voice listen` there: it names WSL2 |
| Doctor lines for the audio server, display and systemd | Verified with fake sockets and directories | `hal-voice doctor` |
| Hermetic test suite | Wired into CI on a Windows runner; not yet seen green | The `wsl` job of the CI workflow |
| Microphone and speakers through WSLg | Unverified | `hal-voice devices`, then `hal-voice listen` |
| Hands-free listening (`activation.keyword`) | Unverified; it needs only the microphone | `hal-voice voicetest`, or say "Hal" to `./hal-voice` |
| Push-to-talk keys (`pynput`) | Unverified; it sees keys only in WSLg windows, not in Windows apps | Focus a WSLg window and press the key |
| The eye window (`hal-voice eye`) | Unverified; needs WSLg | `hal-voice eye` |
| systemd units | Unverified; needs systemd in `/etc/wsl.conf` | `systemctl --user is-system-running` |
| CPU speech and voice (`gpu.mode: on_demand`) | Unverified on WSL2; no GPU needed | `hal-voice tts --out /tmp/test.wav "Hello"` |
| GPU in `resident` mode | Unverified | `hal-voice doctor`, the `GPU sharing` line |
| TTS container | Unverified; needs Docker in the distro | `hal-voice tts-server` |

## Symptoms and fixes

| Symptom | Fix |
| --- | --- |
| `doctor` says `no audio server at /mnt/wslg/PulseServer` | Run `wsl --update` in PowerShell, then `wsl --shutdown` and reopen Ubuntu. |
| `hal-voice devices` lists no devices | Install `libportaudio2` and `pulseaudio-utils`, then check `pactl info`. |
| The microphone is silent | In Windows, open Settings, Privacy, Microphone, and allow desktop apps. Then restart the distro. |
| Audio crackles or stutters | Keep the project off `/mnt/c`, and close other apps that use the audio device (unverified cause). |
| Push-to-talk does nothing | `pynput` sees only WSLg windows. Use hands-free mode (`activation.keyword.enabled: true`). |
| `doctor` warns that there is no display | Run `wsl --update` to get WSLg, then `wsl --shutdown`. |
| `doctor` warns that systemd is not running | Do step 6. |
| `systemctl --user` says `Failed to connect to bus` | Do step 6, or start `./hal-voice` by hand. |
| `hal-voice` says the platform is not supported | You are on native Windows or macOS. Run it inside a WSL2 Ubuntu shell. |
| `tts-server` says docker is not installed | Install Docker Engine in Ubuntu, or turn on WSL integration in Docker Desktop. |
| `tts-server` says docker has no `nvidia` runtime | Install the NVIDIA Container Toolkit in Ubuntu, or set `tts.device: cpu`. |
| Model downloads and the build are slow | Move the checkout from `/mnt/c` into `~`. |
