# GPU container for the GPT-SoVITS api_v2 server (hal-voice tts.server.isolation: container).
# Built by `hal-voice tts-image build`, whose context holds this file, GPT-SoVITS's
# requirements.txt and tts_launcher.py. GPT-SoVITS itself and the weights are mounted
# read-only at run time, so the image carries only the Python environment.
# Unverified: the build needs the network and several GB; check with `hal-voice tts-image build`.
ARG BASE_IMAGE=nvidia/cuda:12.6.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
COPY tts_launcher.py /opt/tts_launcher.py

WORKDIR /opt/gsv
# pickled checkpoints load here because the box is the guard (no network, read-only mounts)
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python3", "/opt/tts_launcher.py"]
