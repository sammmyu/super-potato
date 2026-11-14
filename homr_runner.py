"""Runs homr on a chosen processor. score2xml.py starts this with the Python of the
matching backend environment (.venv, .venv-nvidia or .venv-amd):

    python homr_runner.py --device cpu|nvidia|amd --workdir DIR page1.jpg [page2.jpg ...]
    python homr_runner.py --device nvidia --init     # download models, then test the GPU

Pages are copied into DIR as page01.png, page02.png, ... and homr writes DIR/page01.musicxml.

homr itself knows CUDA (NVIDIA) and ROCm (AMD, Linux only). On Windows, AMD cards
are reached through DirectML instead, so for --device amd this script points homr's
GPU provider selection at DirectML before running it.

Lines starting with "S2X" are read by score2xml.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def prepare_pages(inputs: list[Path], workdir: Path) -> list[Path]:
    """Copy inputs into workdir as page01.png, ... PDFs are passed as-is (homr renders
    them at 300 dpi). Photos are turned upright using their EXIF rotation."""
    from PIL import Image, ImageOps

    pages = []
    for i, src in enumerate(inputs, 1):
        ext = src.suffix.lower()
        if ext == ".pdf":
            dst = workdir / f"page{i:02d}.pdf"
            shutil.copyfile(src, dst)
        elif ext in IMAGE_EXTS:
            dst = workdir / f"page{i:02d}.png"
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im)
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                im.save(dst)
        else:
            raise SystemExit(f"Unsupported file type: {src.name}")
        pages.append(dst)
    return pages


# File-name prefixes of the homr models DirectML runs. The decoder stays on the CPU: it
# makes one tiny call per note, and DirectML's per-call overhead made it 4x slower.
DML_MODELS = ("segnet", "encoder")


def dml_providers() -> list:
    return [("DmlExecutionProvider", {"device_id": int(os.environ.get("S2X_DML_DEVICE", "0"))}),
            "CPUExecutionProvider"]


def patch_sessions(ort, device: str) -> None:
    """Report which processor each model ends up on, and for DirectML put homr's heavy
    models on the GPU. homr runs in CPU mode then (full-precision models, data kept in
    CPU memory) because its half-precision GPU models produce NaNs under DirectML."""
    base = ort.InferenceSession

    class PatchedSession(base):
        def __init__(self, path_or_bytes, *args, **kwargs):
            name = os.path.basename(str(path_or_bytes)) if isinstance(
                path_or_bytes, (str, os.PathLike)) else "model"
            if device == "amd" and name.startswith(DML_MODELS) and not args \
                    and not kwargs.get("providers"):
                kwargs["providers"] = dml_providers()
            super().__init__(path_or_bytes, *args, **kwargs)
            log(f"S2X-SESSION {name} {self.get_providers()[0]}")

    ort.InferenceSession = PatchedSession


def setup(device: str) -> str:
    """Prepare onnxruntime for the device; return homr's --gpu value."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if device == "nvidia":
        if "CUDAExecutionProvider" not in available:
            raise SystemExit("S2X-ERROR This environment has no CUDA support installed.")
        ort.preload_dlls()  # CUDA/cuDNN DLLs that pip installed alongside onnxruntime-gpu
    elif device == "amd":
        if "DmlExecutionProvider" not in available:
            raise SystemExit("S2X-ERROR This environment has no DirectML support installed.")

    patch_sessions(ort, device)
    log(f"S2X-DEVICE {device}")
    return "force" if device == "nvidia" else "no"


def check_gpu(device: str) -> None:
    """Load homr's segmentation model on the GPU to prove the GPU path really works."""
    import onnxruntime as ort

    if device == "nvidia":
        from homr.onnx_providers import gpu_providers
        from homr.segmentation.config import segnet_path_onnx_fp16 as model

        providers = gpu_providers()[0]
    else:
        from homr.segmentation.config import segnet_path_onnx as model

        providers = dml_providers()
    session = ort.InferenceSession(model, providers=providers)
    used = session.get_providers()[0]
    if used == "CPUExecutionProvider":
        raise SystemExit(f"S2X-ERROR The {device} GPU could not be used (fell back to CPU).")
    log(f"S2X-CHECK-OK {used}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "nvidia", "amd"], required=True)
    ap.add_argument("--workdir", type=Path)
    ap.add_argument("--init", action="store_true", help="download models, then test the device")
    ap.add_argument("inputs", nargs="*", type=Path)
    args = ap.parse_args()

    gpu_flag = setup(args.device)
    from homr.main import main as homr_main

    if args.init:
        sys.argv = ["homr", "--gpu", gpu_flag, "--init"]
        homr_main()
        if args.device != "cpu":
            check_gpu(args.device)
        return

    if not args.inputs or not args.workdir:
        raise SystemExit("S2X-ERROR No pages given.")
    pages = prepare_pages(args.inputs, args.workdir)
    os.chdir(args.workdir)
    sys.argv = ["homr", "--gpu", gpu_flag, *[p.name for p in pages]]
    homr_main()


if __name__ == "__main__":
    main()
