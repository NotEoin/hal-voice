"""voicebuild: turns a copy of the film into HAL's voice weights.

Finds every line HAL speaks clearly, cleans it, trains GPT-SoVITS on the result and installs
the best checkpoint as hal-voice's voice. Run it with ./make-hal-voice; see docs/howto/build-the-voice.md.
"""
