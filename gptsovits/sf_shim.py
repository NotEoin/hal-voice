"""Runs a GPT-SoVITS script with torchaudio.load backed by soundfile.

Recent torchaudio decodes through torchcodec, which needs NVIDIA's NPP libraries;
the training clips are plain WAV, so soundfile reads them without that stack.

    python sf_shim.py SCRIPT [ARGS ...]
"""

import runpy
import sys

import soundfile
import torch
import torchaudio


def _load(path, *args, **kwargs):
    data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
