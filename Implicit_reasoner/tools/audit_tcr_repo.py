import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEY_FILES = [
    "dataset/tcr_video_sampling.py",
    "dataset/it_dataset_mistral.py",
    "models/videochat_mistra/videochat2_it_mistral.py",
    "inference_icr_mc_tcr.py",
    "inference_icr_open_tcr.py",
    "tools/score_mc.py",
    "tools/check_no_duration_leak.py",
]


def fail(msg):
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def main():
    for rel in KEY_FILES:
        if not (ROOT / rel).exists():
            fail(f"required file missing: {rel}")

    banned = ["/mnt/sdb", "/home/next", "training_5k.json", "pred = 'A'", "unclear from visible context"]
    for rel in ["dataset/it_dataset_mistral.py", "models/videochat_mistra/videochat2_it_mistral.py", "inference_icr_mc_tcr.py", "inference_icr_open_tcr.py"]:
        text = (ROOT / rel).read_text()
        for b in banned:
            if b in text:
                fail(f"banned pattern found in {rel}: {b}")

    subprocess.check_call([sys.executable, "-m", "py_compile", *[str(ROOT / f) for f in KEY_FILES]])

    mc_text = (ROOT / "inference_icr_mc_tcr.py").read_text()
    if "VideoChat2_it_mistral" not in mc_text or "sample_tcr_multi_views" not in mc_text:
        fail("mc inference missing model/tcr imports")
    op_text = (ROOT / "inference_icr_open_tcr.py").read_text()
    if "VideoChat2_it_mistral" not in op_text or "sample_tcr_multi_views" not in op_text:
        fail("open inference missing model/tcr imports")

    ds_text = (ROOT / "dataset/it_dataset_mistral.py").read_text()
    if "sample_tcr_frame_indices" not in ds_text and "sample_tcr_multi_views" not in ds_text:
        fail("dataset missing tcr sampler use")

    ast.parse((ROOT / "inference_icr_mc_tcr.py").read_text())
    ast.parse((ROOT / "inference_icr_open_tcr.py").read_text())
    print("[OK] audit passed")


if __name__ == "__main__":
    main()
