#!/usr/bin/env python3
"""Headless HAL fine-tune driver for GPT-SoVITS (v2).

Replicates exactly what the GPT-SoVITS WebUI does for:
  1A  text + BERT feature extraction   (1-get-text.py)
  1B  HuBERT/wav32k feature extraction  (2-get-hubert-wav32k.py)
  1C  semantic-token extraction         (3-get-semantic.py)
  then SoVITS training (s2_train.py) and GPT training (s1_train.py).

Run with the gptsovits conda python, from anywhere -- it cd's into the GPT-SoVITS dir.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------- paths / config
# Resolve everything relative to this file so the repo works wherever it's cloned.
HERE = Path(__file__).resolve().parent  # <repo>/gptsovits
ROOT = HERE.parent  # <repo>
GSV = Path(os.environ.get("GPTSOVITS_DIR", HERE / "GPT-SoVITS"))
VOICE = ROOT / "voice-data"
WEIGHTS_OUT = HERE / "weights"
os.chdir(GSV)
PY = sys.executable

# The prepare/train scripts import `tools.*` (repo root) and `text.*`/`module.*`/`AR.*`
# (the GPT_SoVITS package dir). Both must be on PYTHONPATH for the subprocesses.
os.environ["PYTHONPATH"] = f"{GSV}:{GSV / 'GPT_SoVITS'}:" + os.environ.get("PYTHONPATH", "")

# torch >= 2.6 defaults torch.load(weights_only=True), which rejects the pretrained
# GPT checkpoint (it pickles a pathlib.PosixPath). These are trusted local files --
# force the legacy full-unpickle behaviour so loading succeeds.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# A new name keeps a retrain from overwriting the weights in use.
EXP = os.environ.get("HAL_EXP", "hal")
# v2ProPlus conditions the decoder on a speaker embedding of the reference, which
# holds timbre far better than v2; set HAL_VERSION=v2 for the older, lighter model.
VERSION = os.environ.get("HAL_VERSION", "v2")
PRO = VERSION in ("v2Pro", "v2ProPlus")
EXP_ROOT = "logs"
OPT_DIR = f"{EXP_ROOT}/{EXP}"
LIST = str((VOICE / "clean" / "hal.list").resolve())
WAVDIR = str((VOICE / "clean" / "slices").resolve())

BERT = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
HUBERT = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
SV_PATH = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
S2G = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
S2D = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
S1 = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
if PRO:
    S2G = f"GPT_SoVITS/pretrained_models/v2Pro/s2G{VERSION}.pth"
    S2D = f"GPT_SoVITS/pretrained_models/v2Pro/s2D{VERSION}.pth"
    S1 = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
S2_CONFIG = f"GPT_SoVITS/configs/s2{VERSION}.json" if PRO else "GPT_SoVITS/configs/s2.json"
SOVITS_DIR = f"SoVITS_weights_{VERSION}"
GPT_DIR = f"GPT_weights_{VERSION}"

# Single-speaker character voice. train-hal.md: SoVITS ~8 epochs, GPT ~15 epochs.
# GPT lowered to 12 -- fewer epochs = smoother/legato (more epochs -> stiff, robotic).
SOVITS_EPOCHS = int(os.environ.get("HAL_SOVITS_EPOCHS", 8))
SOVITS_SAVE_EVERY = 4
SOVITS_BATCH = 4
GPT_EPOCHS = int(os.environ.get("HAL_GPT_EPOCHS", 12))
GPT_SAVE_EVERY = 4  # divides 12 so the final e12 weight is saved
GPT_BATCH = 4

os.makedirs(OPT_DIR, exist_ok=True)
os.makedirs("TEMP", exist_ok=True)
# GPT-SoVITS saves trained weights here but does not create the dirs itself.
os.makedirs(SOVITS_DIR, exist_ok=True)
os.makedirs(GPT_DIR, exist_ok=True)
WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)


def run_step(script, extra_env, shim=False):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in extra_env.items()})
    print(f"\n=== RUN {script} ===", flush=True)
    cmd = [PY, "-s", str(HERE / "sf_shim.py"), script] if shim else [PY, "-s", script]
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        sys.exit(f"FAILED: {script} (exit {r.returncode})")


def base_env():
    return {
        "inp_text": LIST,
        "inp_wav_dir": WAVDIR,
        "exp_name": EXP,
        "opt_dir": OPT_DIR,
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "is_half": "True",
        "version": VERSION,
    }


# ============================================================ 1A: text + BERT
path_text = f"{OPT_DIR}/2-name2text.txt"
if not (
    os.path.exists(path_text)
    and len(open(path_text, encoding="utf8").read().strip().split("\n")) >= 2
):
    run_step(
        "GPT_SoVITS/prepare_datasets/1-get-text.py", {**base_env(), "bert_pretrained_dir": BERT}
    )
    opt = []
    p0 = f"{OPT_DIR}/2-name2text-0.txt"
    with open(p0, encoding="utf8") as f:
        opt += f.read().strip("\n").split("\n")
    os.remove(p0)
    with open(path_text, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
    if not "".join(opt):
        sys.exit("1A produced no text")
print("1A done")

# ============================================================ 1B: HuBERT / wav32k
run_step(
    "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
    {**base_env(), "cnhubert_base_dir": HUBERT, "sv_path": SV_PATH},
)
if PRO:
    run_step(
        "GPT_SoVITS/prepare_datasets/2-get-sv.py", {**base_env(), "sv_path": SV_PATH}, shim=True
    )
print("1B done")

# ============================================================ 1C: semantic tokens
path_semantic = f"{OPT_DIR}/6-name2semantic.tsv"
if not (os.path.exists(path_semantic) and os.path.getsize(path_semantic) >= 31):
    run_step(
        "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
        {**base_env(), "pretrained_s2G": S2G, "s2config_path": S2_CONFIG},
    )
    opt = ["item_name\tsemantic_audio"]
    sp = f"{OPT_DIR}/6-name2semantic-0.tsv"
    with open(sp, encoding="utf8") as f:
        opt += f.read().strip("\n").split("\n")
    os.remove(sp)
    with open(path_semantic, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
print("1C done")

# ============================================================ SoVITS training (s2)
s2_dir = f"{EXP_ROOT}/{EXP}"
# Clear stale step-checkpoints so training runs the full epoch count fresh
# (GPT-SoVITS auto-resumes from G_*/D_* and would otherwise skip past our target).
shutil.rmtree(f"{s2_dir}/logs_s2_{VERSION}", ignore_errors=True)
os.makedirs(f"{s2_dir}/logs_s2_{VERSION}", exist_ok=True)
with open(S2_CONFIG) as f:
    s2 = json.load(f)
s2["train"]["batch_size"] = SOVITS_BATCH
s2["train"]["epochs"] = SOVITS_EPOCHS
s2["train"]["text_low_lr_rate"] = 0.4
s2["train"]["pretrained_s2G"] = S2G
s2["train"]["pretrained_s2D"] = S2D
s2["train"]["if_save_latest"] = True
s2["train"]["if_save_every_weights"] = True
s2["train"]["save_every_epoch"] = SOVITS_SAVE_EVERY
s2["train"]["gpu_numbers"] = "0"
s2["train"]["grad_ckpt"] = False
s2["train"]["lora_rank"] = 32
s2["model"]["version"] = VERSION
s2["data"]["exp_dir"] = s2["s2_ckpt_dir"] = s2_dir
s2["save_weight_dir"] = SOVITS_DIR
s2["name"] = EXP
s2["version"] = VERSION
tmp_s2 = "TEMP/tmp_s2.json"
with open(tmp_s2, "w") as f:
    json.dump(s2, f)
print("\n=== TRAIN SoVITS (s2) ===", flush=True)
r = subprocess.run([PY, "-s", "GPT_SoVITS/s2_train.py", "--config", tmp_s2])
if r.returncode != 0:
    sys.exit(f"SoVITS training failed (exit {r.returncode})")

# ============================================================ GPT training (s1)
s1_dir = f"{EXP_ROOT}/{EXP}"
shutil.rmtree(f"{s1_dir}/logs_s1_{VERSION}", ignore_errors=True)
os.makedirs(f"{s1_dir}/logs_s1", exist_ok=True)
with open("GPT_SoVITS/configs/s1longer-v2.yaml") as f:
    s1 = yaml.safe_load(f)
s1["train"]["batch_size"] = GPT_BATCH
s1["train"]["epochs"] = GPT_EPOCHS
s1["pretrained_s1"] = S1
s1["train"]["save_every_n_epoch"] = GPT_SAVE_EVERY
s1["train"]["if_save_every_weights"] = True
s1["train"]["if_save_latest"] = True
s1["train"]["if_dpo"] = False
s1["train"]["half_weights_save_dir"] = GPT_DIR
s1["train"]["exp_name"] = EXP
s1["train_semantic_path"] = f"{s1_dir}/6-name2semantic.tsv"
s1["train_phoneme_path"] = f"{s1_dir}/2-name2text.txt"
s1["output_dir"] = f"{s1_dir}/logs_s1_{VERSION}"
os.environ["_CUDA_VISIBLE_DEVICES"] = "0"
os.environ["hz"] = "25hz"
tmp_s1 = "TEMP/tmp_s1.yaml"
with open(tmp_s1, "w") as f:
    yaml.dump(s1, f, default_flow_style=False)
print("\n=== TRAIN GPT (s1) ===", flush=True)
r = subprocess.run([PY, "-s", "GPT_SoVITS/s1_train.py", "--config_file", tmp_s1])
if r.returncode != 0:
    sys.exit(f"GPT training failed (exit {r.returncode})")


# ============================================================ collect weights
def newest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


sovits_w = newest(f"{SOVITS_DIR}/{EXP}_*.pth")
gpt_w = newest(f"{GPT_DIR}/{EXP}-*.ckpt")
print("\n=== RESULT ===")
print("SoVITS weight:", sovits_w)
print("GPT weight   :", gpt_w)
if sovits_w:
    shutil.copy2(sovits_w, WEIGHTS_OUT / os.path.basename(sovits_w))
if gpt_w:
    shutil.copy2(gpt_w, WEIGHTS_OUT / os.path.basename(gpt_w))
print("Copied weights to", WEIGHTS_OUT)
# Write the resolved names so the launcher can patch config.yaml.
with open(WEIGHTS_OUT / "TRAINED.txt", "w") as f:
    f.write(f"sovits={os.path.basename(sovits_w) if sovits_w else ''}\n")
    f.write(f"gpt={os.path.basename(gpt_w) if gpt_w else ''}\n")
print("DONE")
