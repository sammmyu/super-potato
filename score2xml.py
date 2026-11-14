"""Turn a picture (or PDF) of sheet music into a MusicXML file that Sibelius 7 can open.

Recognition is done by homr (https://github.com/liebharc/homr), an open-source
optical music recognition engine, on the CPU, an NVIDIA GPU (CUDA) or an AMD GPU
(DirectML). Each processor needs a different build of onnxruntime and those can't
share one Python environment, so each lives in its own folder (.venv, .venv-nvidia,
.venv-amd); homr_runner.py runs inside it.

homr writes MusicXML 4.0 with some elements out of order. Sibelius 7 reads MusicXML 2.0
and refuses files that don't follow its DTD, so the result is rewritten as valid
MusicXML 2.0 (see musicxml20.py) and saved as an uncompressed .xml file.

This file only needs the Python standard library.

Usage:
    python score2xml.py page1.jpg [page2.jpg ...] [-o out.xml] [--title T] [--composer C]
                        [--device auto|cpu|nvidia|amd]
    python score2xml.py --install nvidia      (or cpu / amd)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import musicxml20

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "homr_runner.py"
READY_MARKER = "s2x-ready.txt"

ANSI_COLOUR = re.compile(r"\x1b?\[[0-9;]*m")

INPUT_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".pdf"}

Log = Callable[[str], None]


@dataclass(frozen=True)
class Backend:
    key: str
    label: str
    folder: str
    packages: tuple[str, ...]
    size: str  # disk space once installed, shown in the GUI


BACKENDS = {
    "cpu": Backend("cpu", "CPU", ".venv",
                   ("homr[cpu]==0.7.1", "tkinterdnd2==0.7.0"), "about 0.5 GB"),
    "nvidia": Backend("nvidia", "NVIDIA GPU (CUDA)", ".venv-nvidia",
                      ("homr[cuda]==0.7.1",), "about 2.3 GB"),
    "amd": Backend("amd", "AMD GPU (DirectML)", ".venv-amd",
                   ("homr==0.7.1", "onnxruntime-directml==1.24.4"), "about 0.6 GB"),
}

DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 2.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)

# Child order required by the MusicXML 2.0 DTD, for the elements homr gets wrong
# (or could plausibly get wrong). Children not listed keep their relative order
# and go to the end.
CHILD_ORDER = {
    "score-partwise": ["work", "movement-number", "movement-title", "identification",
                       "defaults", "credit", "part-list", "part"],
    "identification": ["creator", "rights", "encoding", "source", "relation", "miscellaneous"],
    "score-part": ["identification", "part-name", "part-name-display", "part-abbreviation",
                   "part-abbreviation-display", "group", "score-instrument",
                   "midi-device", "midi-instrument"],
    "score-instrument": ["instrument-name", "instrument-abbreviation", "instrument-sound",
                         "solo", "ensemble", "virtual-instrument"],
    "midi-instrument": ["midi-channel", "midi-name", "midi-bank", "midi-program",
                        "midi-unpitched", "volume", "pan", "elevation"],
    "attributes": ["footnote", "level", "divisions", "key", "time", "staves", "part-symbol",
                   "instruments", "clef", "staff-details", "transpose", "directive",
                   "measure-style"],
    "note": ["grace", "cue", "chord", "pitch", "unpitched", "rest", "duration", "tie",
             "instrument", "footnote", "level", "voice", "type", "dot", "accidental",
             "time-modification", "stem", "notehead", "notehead-text", "staff", "beam",
             "notations", "lyric", "play"],
    "pitch": ["step", "alter", "octave"],
    "unpitched": ["display-step", "display-octave"],
    "rest": ["display-step", "display-octave"],
    "direction": ["direction-type", "offset", "footnote", "level", "voice", "staff", "sound"],
    "barline": ["bar-style", "footnote", "level", "wavy-line", "segno", "coda", "fermata",
                "ending", "repeat"],
    "backup": ["duration", "footnote", "level"],
    "forward": ["duration", "footnote", "level", "voice", "staff"],
}

# Messages homr prints when it gives up on the GPU and carries on with the CPU.
GPU_FALLBACK_HINTS = ("error while trying to load model on the gpu", "going on without gpu",
                      "falling back to cpu", "no gpu execution provider available")


class Cancelled(Exception):
    pass


class ProcessHolder:
    """Lets another thread (the GUI) cancel whatever subprocess is running."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            if os.name == "nt":  # the venv's python.exe is a launcher; kill its child too
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                               capture_output=True, creationflags=_no_window())
            self.proc.kill()


# --------------------------------------------------------------------------- backends

def backend_python(key: str) -> Path:
    folder = HERE / BACKENDS[key].folder
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def backend_ready(key: str) -> bool:
    return backend_python(key).exists() and (HERE / BACKENDS[key].folder / READY_MARKER).exists()


def _no_window() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def run_streaming(cmd: list[str], log: Log, holder: ProcessHolder | None = None,
                  cwd: Path | None = None) -> tuple[int, list[str]]:
    """Run cmd, passing each output line to log as it arrives. Returns (exit code, lines)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", creationflags=_no_window())
    if holder:
        holder.proc = proc
        if holder.cancelled:
            proc.kill()
    lines = []
    assert proc.stdout
    for line in proc.stdout:  # text mode splits on \r too, so progress bars come line by line
        # onnxruntime's native warnings arrive as UTF-16 with colour codes; tidy them.
        line = ANSI_COLOUR.sub("", line.replace("\x00", "")).rstrip()
        if line:
            lines.append(line)
            log(line)
    code = proc.wait()
    if holder and holder.cancelled:
        raise Cancelled()
    return code, lines


def base_python() -> str:
    """A real (non-venv) python.exe to create new environments with."""
    exe = Path(getattr(sys, "_base_executable", sys.executable))
    candidate = exe.with_name("python.exe" if os.name == "nt" else "python3")
    return str(candidate if candidate.exists() else exe)


def install_backend(key: str, log: Log, holder: ProcessHolder | None = None) -> None:
    """Create the backend's environment, install packages, download models, test it."""
    b = BACKENDS[key]
    folder = HERE / b.folder
    py = backend_python(key)
    (folder / READY_MARKER).unlink(missing_ok=True)

    steps = []
    if not py.exists():
        steps.append(("Creating Python environment", [base_python(), "-m", "venv", str(folder)]))
    steps += [
        ("Installing packages (this is the slow part)",
         [str(py), "-m", "pip", "install", "--disable-pip-version-check", *b.packages]),
        ("Downloading models and testing", [str(py), str(RUNNER), "--device", key, "--init"]),
    ]
    for title, cmd in steps:
        log(f"== {title}")
        code, lines = run_streaming(cmd, log, holder)
        if code != 0:
            errors = [ln.removeprefix("S2X-ERROR ") for ln in lines if ln.startswith("S2X-ERROR")]
            raise RuntimeError(errors[-1] if errors else
                               f"{title} failed (exit code {code}). See the log for details.")
    (folder / READY_MARKER).write_text(" ".join(b.packages) + "\n", encoding="utf-8")
    log(f"== {b.label} is ready")


def detect_gpus() -> list[str]:
    """Names of the graphics adapters Windows knows about."""
    if os.name != "nt":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"],
            capture_output=True, text=True, timeout=20, creationflags=_no_window()).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def gpu_vendor_present(gpus: list[str], key: str) -> bool:
    names = " ".join(gpus).lower()
    if key == "nvidia":
        return "nvidia" in names
    if key == "amd":
        return "amd" in names or "radeon" in names
    return True


def auto_device(gpus: list[str] | None = None) -> str:
    """Fastest installed backend that has matching hardware."""
    gpus = detect_gpus() if gpus is None else gpus
    if backend_ready("nvidia") and gpu_vendor_present(gpus, "nvidia"):
        return "nvidia"
    if backend_ready("amd") and gpu_vendor_present(gpus, "amd"):
        return "amd"
    return "cpu"


# --------------------------------------------------------------------------- clean-up

def reorder_children(el: ET.Element, order: list[str]) -> None:
    rank = {tag: i for i, tag in enumerate(order)}
    kids = list(el)
    ordered = sorted(kids, key=lambda c: rank.get(c.tag, len(order)))
    if ordered != kids:
        for c in kids:
            el.remove(c)
        el.extend(ordered)


def merge_attributes(measure: ET.Element) -> None:
    """Fold back-to-back <attributes> blocks into one (later values win), so Sibelius
    doesn't see e.g. two different time signatures at the same spot."""
    kids = list(measure)
    i = 0
    while i < len(kids) - 1:
        a, b = kids[i], kids[i + 1]
        if a.tag == b.tag == "attributes":
            for child in list(b):
                same = [c for c in a if c.tag == child.tag
                        and c.get("number") == child.get("number")
                        and child.tag not in ("footnote", "directive", "staff-details")]
                for c in same:
                    a.remove(c)
                a.append(child)
            measure.remove(b)
            kids.pop(i + 1)
        else:
            i += 1


def set_text(parent: ET.Element, path: list[str], text: str, attrib: dict | None = None) -> None:
    el = parent
    for j, tag in enumerate(path):
        match = None
        for c in el.findall(tag):
            if j < len(path) - 1 or not attrib or all(c.get(k) == v for k, v in attrib.items()):
                match = c
                break
        if match is None:
            match = ET.SubElement(el, tag, attrib if j == len(path) - 1 and attrib else {})
        el = match
    el.text = text


def _drop_if(parent: ET.Element, child: ET.Element, condition: bool) -> None:
    if condition:
        parent.remove(child)


def downgrade_to_2_0(root: ET.Element) -> None:
    """Remove what MusicXML 2.0 doesn't have: elements and attributes added in 3.0-4.0
    (musicxml20.py), plus the value-level rules of the official 3.0-to-2.0 stylesheet
    (to20.xsl in the MusicXML 4.0 schema folder)."""
    continuing = ("wedge", "dashes", "bracket", "octave-shift", "pedal")
    for parent in list(root.iter()):
        for child in list(parent):
            tag = child.tag
            if tag in musicxml20.NEW_ELEMENTS:
                parent.remove(child)
                continue
            for attr in musicxml20.NEW_ATTRIBUTES.get(tag, ()):
                child.attrib.pop(attr, None)
            text = (child.text or "").strip()
            _drop_if(parent, child,
                     (tag == "tied" and child.get("type") == "continue")
                     or (tag in ("beam", "metronome-beam") and child.get("number") in ("7", "8"))
                     or (tag in ("part-symbol", "group-symbol") and text == "square")
                     or (tag == "tremolo" and text in ("7", "8"))
                     or (tag == "clef" and child.findtext("sign") == "jianpu")
                     or (tag == "direction" and any(
                         e.get("type") == "continue"
                         for name in continuing for e in child.iterfind(f"direction-type/{name}"))))
            if tag == "time" and child.get("symbol") in ("note", "dotted-note"):
                del child.attrib["symbol"]
            if tag == "cancel" and child.get("location") == "before-barline":
                del child.attrib["location"]
        # 2.0 allows only one <transpose> per <attributes> and one <midi-device> per part.
        for name in ("transpose", "midi-device"):
            for extra in parent.findall(name)[1:]:
                parent.remove(extra)

    # Containers that removals may have emptied.
    for _ in range(3):
        for parent in list(root.iter()):
            for child in list(parent):
                if child.tag in ("notations", "dynamics", "direction-type", "articulations",
                                 "ornaments", "technical") and len(child) == 0:
                    parent.remove(child)
                elif child.tag == "direction" and child.find("direction-type") is None:
                    parent.remove(child)


def clean_for_sibelius(root: ET.Element, title: str | None, composer: str | None) -> None:
    if root.tag != "score-partwise":
        raise RuntimeError(f"Unexpected MusicXML root <{root.tag}>")
    root.set("version", "2.0")

    # homr's <defaults> page sizes are placeholders; let Sibelius use its house style.
    for d in root.findall("defaults"):
        root.remove(d)

    guessed = (root.findtext("work/work-title") or "").strip()
    if title is None and "=" in guessed and any(ch.isdigit() for ch in guessed):
        title = ""  # homr read a tempo mark such as "♩.=60" as the title

    if title is not None:
        for w in root.findall("work"):
            root.remove(w)
        for mt in root.findall("movement-title"):
            root.remove(mt)
        if title:
            set_text(root, ["work", "work-title"], title)
    if composer:
        set_text(root, ["identification", "creator"], composer, {"type": "composer"})

    downgrade_to_2_0(root)

    for measure in root.iter("measure"):
        merge_attributes(measure)

    for note in root.iter("note"):
        if note.find("grace") is not None:  # grace notes may not have a duration
            for d in note.findall("duration"):
                note.remove(d)

    for el in root.iter():
        order = CHILD_ORDER.get(el.tag)
        if order:
            reorder_children(el, order)


def check_measures(root: ET.Element) -> list[str]:
    """Return measure numbers whose contents don't add up to the time signature."""
    bad = []
    for part in root.findall("part"):
        divisions, expected = 1, None
        measures = part.findall("measure")
        for idx, m in enumerate(measures):
            for a in m.findall("attributes"):
                if a.findtext("divisions"):
                    divisions = int(a.findtext("divisions"))
                t = a.find("time")
                if t is not None:
                    try:
                        expected = Fraction(int(t.findtext("beats")), int(t.findtext("beat-type"))) * 4
                    except (TypeError, ValueError):
                        expected = None
            if expected is None or m.get("implicit") == "yes":
                continue
            pos = longest = 0
            for c in m:
                if c.tag == "note" and c.find("chord") is None and c.find("grace") is None:
                    pos += int(c.findtext("duration") or 0)
                elif c.tag == "forward":
                    pos += int(c.findtext("duration") or 0)
                elif c.tag == "backup":
                    pos -= int(c.findtext("duration") or 0)
                longest = max(longest, pos)
            length = Fraction(longest, divisions)
            is_edge = idx in (0, len(measures) - 1)  # pick-up / final bar may be short
            if length > expected or (length < expected and not is_edge):
                bad.append(m.get("number", str(idx + 1)))
    return bad


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    path.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n{DOCTYPE}\n{body}\n',
                    encoding="utf-8")


# --------------------------------------------------------------------------- conversion

@dataclass
class Result:
    output: Path
    bars: int
    notes: int
    bad_bars: list[str]
    seconds: float
    device: str
    warnings: list[str] = field(default_factory=list)


def free_name(path: Path) -> Path:
    if not path.exists():
        return path
    n = 2
    while (candidate := path.with_name(f"{path.stem} ({n}){path.suffix}")).exists():
        n += 1
    return candidate


def default_output(first_input: Path) -> Path:
    return free_name(first_input.resolve().with_suffix(".xml"))


def convert(inputs: list[Path], output: Path | None = None, title: str | None = None,
            composer: str | None = None, device: str = "auto", log: Log = print,
            holder: ProcessHolder | None = None) -> Result:
    """Recognise the pages (joined in order into one score) and write Sibelius-ready MusicXML."""
    if not inputs:
        raise ValueError("No pages given.")
    for p in inputs:
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix.lower() not in INPUT_EXTS:
            raise ValueError(f"Unsupported file type: {p.name} "
                             f"(use {', '.join(sorted(INPUT_EXTS))})")
    if device == "auto":
        device = auto_device()
    if not backend_ready(device):
        raise RuntimeError(f"{BACKENDS[device].label} isn't installed yet "
                           f"(run: python score2xml.py --install {device}).")

    out = (output or default_output(inputs[0])).resolve()
    if out.suffix.lower() != ".xml":
        out = out.with_suffix(".xml")

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="score2xml_") as tmp:
        cmd = [str(backend_python(device)), str(RUNNER), "--device", device, "--workdir", tmp,
               *[str(p.resolve()) for p in inputs]]
        code, lines = run_streaming(cmd, log, holder)
        raw = Path(tmp) / "page01.musicxml"
        if code != 0 or not raw.exists():
            reasons = [ln.removeprefix("S2X-ERROR ") for ln in lines
                       if "error" in ln.lower() and not ln.startswith("S2X-SESSION")]
            raise RuntimeError("homr could not read the score.\n"
                               + ("\n".join(reasons[-3:]) or f"(exit code {code})")
                               + "\nTry a sharper, straighter, well-lit image.")
        root = ET.parse(raw).getroot()

    warnings = []
    lowered = [ln.lower() for ln in lines]
    # Models meant to run on the GPU (title OCR always runs on the CPU; with DirectML
    # so does the decoder, see homr_runner.py).
    gpu_models = ("segnet", "encoder") + (("decoder",) if device == "nvidia" else ())
    sessions = [ln.split()[1:3] for ln in lines if ln.startswith("S2X-SESSION ")]
    on_cpu = [name for name, provider in sessions
              if name.startswith(gpu_models) and provider == "CPUExecutionProvider"]
    if device != "cpu" and (on_cpu or any(h in ln for ln in lowered for h in GPU_FALLBACK_HINTS)):
        warnings.append(f"The {BACKENDS[device].label} couldn't be used for every step, "
                        "so part of the work ran on the CPU.")

    clean_for_sibelius(root, title, composer)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_xml(root, out)
    return Result(output=out,
                  bars=len(root.findall("part/measure")),
                  notes=sum(1 for n in root.iter("note") if n.find("rest") is None),
                  bad_bars=check_measures(root),
                  seconds=time.time() - started,
                  device=device,
                  warnings=warnings)


# --------------------------------------------------------------------------- CLI

def pause_if_dragged_and_dropped() -> None:
    """When the .bat was started from Explorer (drag-and-drop or double-click), keep the
    console window open so the result can be read. Do nothing in a terminal."""
    # Set by score2xml.bat. Explorer starts batch files with "cmd.exe /c"; a cmd window
    # the user typed into has no /c.
    if os.name != "nt" or " /c " not in os.environ.get("S2X_CMDLINE", "").lower():
        return
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.windll.kernel32
    pids = (wintypes.DWORD * 64)()
    count = k32.GetConsoleProcessList(pids, 64)
    for pid in pids[:min(count, 64)]:
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            continue
        buf, size = ctypes.create_unicode_buffer(260), wintypes.DWORD(260)
        ok = k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        k32.CloseHandle(handle)
        if ok and Path(buf.value).name.lower() not in ("cmd.exe", "python.exe", "py.exe"):
            return  # a shell (PowerShell, bash, ...) owns this console
    input("\nPress Enter to close this window...")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert a picture or PDF of sheet music to MusicXML for Sibelius 7.")
    ap.add_argument("inputs", nargs="*", type=Path,
                    help="image(s) or PDF; several files are joined in order into one score")
    ap.add_argument("-o", "--output", type=Path,
                    help="output .xml file (default: next to the first input)")
    ap.add_argument("--title", help="score title (homr guesses one; pass '' for none)")
    ap.add_argument("--composer", help="composer name")
    ap.add_argument("--device", choices=["auto", *BACKENDS], default="auto",
                    help="processor to use (default: fastest one installed)")
    ap.add_argument("--install", choices=list(BACKENDS), metavar="{cpu,nvidia,amd}",
                    help="set up a processor backend and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="show homr's own output")
    args = ap.parse_args()

    if args.install:
        install_backend(args.install, print)
        return 0
    if not args.inputs:
        ap.error("give at least one image or PDF")

    device = auto_device() if args.device == "auto" else args.device
    print(f"Reading {len(args.inputs)} file(s) on {BACKENDS[device].label}...", flush=True)
    quiet = lambda line: None  # noqa: E731
    try:
        r = convert(args.inputs, args.output, args.title, args.composer, device,
                    log=print if args.verbose else quiet)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e))

    print(f"Done in {r.seconds:.0f}s: {r.bars} bars, {r.notes} notes.")
    print(f"Saved: {r.output}")
    for w in r.warnings:
        print(f"Note: {w}")
    if r.bad_bars:
        shown = ", ".join(r.bad_bars[:20]) + (" ..." if len(r.bad_bars) > 20 else "")
        print(f"Check these bars in Sibelius - their rhythms don't add up: {shown}")
    print("Open it in Sibelius 7 with File > Open (choose 'MusicXML files').")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int) and e.code:
            print(e.code, file=sys.stderr)
    pause_if_dragged_and_dropped()
    sys.exit(code)
