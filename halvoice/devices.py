"""List audio devices so you can pin `audio.input_device` / `audio.output_device`."""

from __future__ import annotations


def main() -> None:
    """Prints the audio input and output devices."""
    try:
        import sounddevice as sd
    except Exception as e:
        raise SystemExit(f"sounddevice not available: {e}\nRun ./setup-voice.sh first.") from e

    print("Audio devices (index: name  [in/out channels]  @default_samplerate):\n")
    default_in, default_out = sd.default.device
    for i, d in enumerate(sd.query_devices()):
        tag = []
        if i == default_in:
            tag.append("DEFAULT-IN")
        if i == default_out:
            tag.append("DEFAULT-OUT")
        flag = ("  <" + ",".join(tag) + ">") if tag else ""
        print(
            f"{i:3d}: {d['name']}  "
            f"[in {d['max_input_channels']} / out {d['max_output_channels']}]  "
            f"@{int(d['default_samplerate'])}{flag}"
        )
    print(
        "\nSet audio.input_device / audio.output_device in config.yaml to one of the indices above."
    )


if __name__ == "__main__":
    main()
