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
import math
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


# Length of each note value, in crotchets.
TYPE_QUARTERS = {"breve": Fraction(8), "whole": Fraction(4), "half": Fraction(2),
                 "quarter": Fraction(1), "eighth": Fraction(1, 2), "16th": Fraction(1, 4),
                 "32nd": Fraction(1, 8), "64th": Fraction(1, 16), "128th": Fraction(1, 32),
                 "256th": Fraction(1, 64)}


def printed_length(note: ET.Element) -> Fraction | None:
    """Length in crotchets implied by a note's printed value, dots and tuplet."""
    base = TYPE_QUARTERS.get(note.findtext("type") or "")
    if base is None:
        return None
    length = base * (2 - Fraction(1, 2 ** len(note.findall("dot"))))
    tm = note.find("time-modification")
    if tm is not None:
        try:
            length = length * int(tm.findtext("normal-notes")) / int(tm.findtext("actual-notes"))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return length


def normalise_durations(root: ET.Element) -> None:
    """Fix bars whose durations were all written in the wrong unit, then give each part
    a single <divisions> value. homr reads each page with its own unit and sometimes
    misses a change when it joins pages: in a trio's cello part every note from page 4
    on lasted half its printed value. A bar is only rescaled when every note in it is
    off by the same factor, so genuine misreadings are left alone (and still flagged)."""
    for part in root.findall("part"):
        divisions = 1
        timed: list[tuple[ET.Element, Fraction]] = []  # every <duration>'s owner, in crotchets
        for m in part.findall("measure"):
            in_bar: list[tuple[ET.Element, Fraction]] = []
            for c in m:
                if c.tag == "attributes" and c.findtext("divisions"):
                    divisions = int(c.findtext("divisions"))
                elif c.tag in ("note", "backup", "forward") and c.findtext("duration"):
                    in_bar.append((c, Fraction(int(c.findtext("duration")), divisions)))
            ratios = {written / printed for c, written in in_bar if c.tag == "note"
                      and (printed := printed_length(c)) and written}
            if len(ratios) == 1 and (k := ratios.pop()) != 1:
                in_bar = [(c, length / k) for c, length in in_bar]
            timed += in_bar
        if not timed:
            continue

        unit = math.lcm(*(length.denominator for _, length in timed))
        for c, length in timed:
            c.find("duration").text = str(int(length * unit))
        for m in part.findall("measure"):
            for a in m.findall("attributes"):
                for d in a.findall("divisions"):
                    a.remove(d)
                if len(a) == 0:
                    m.remove(a)
        first = part.find("measure")
        attributes = first.find("attributes")
        if attributes is None:
            attributes = ET.Element("attributes")
            first.insert(0, attributes)
        ET.SubElement(attributes, "divisions").text = str(unit)


# Number of beams each note value gets.
BEAM_COUNT = {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4, "128th": 5, "256th": 6}


def beam_group_length(beats: int, beat_type: int) -> Fraction:
    """Length of one beam group, in quarter notes, for a time signature."""
    if beat_type == 8 and beats % 3 == 0:
        return Fraction(3, 2)  # 3/8, 6/8, 9/8, 12/8: beam in dotted crotchets
    if beat_type == 2:
        return Fraction(2)  # 2/2, 3/2: beam in minims
    return max(Fraction(4, beat_type), Fraction(1))  # beam per beat, at least a crotchet


def _beam_run(run: list[ET.Element]) -> None:
    """Add <beam> elements to a run of 2+ consecutive beamable notes."""
    levels = [BEAM_COUNT[n.findtext("type")] for n in run]
    for level in range(1, max(levels) + 1):
        i = 0
        while i < len(run):
            if levels[i] < level:
                i += 1
                continue
            j = i
            while j + 1 < len(run) and levels[j + 1] >= level:
                j += 1
            if j > i:
                for k in range(i, j + 1):
                    value = "begin" if k == i else "end" if k == j else "continue"
                    ET.SubElement(run[k], "beam", number=str(level)).text = value
            else:  # a lone semiquaver next to quavers gets a stub ("hook")
                value = "forward hook" if i == 0 else "backward hook"
                ET.SubElement(run[i], "beam", number=str(level)).text = value
            i = j + 1


def add_beams(root: ET.Element) -> None:
    """Join quavers and shorter notes with beams. homr doesn't output beaming, and
    Sibelius keeps whatever the file says, so without this every quaver has its own
    flag. Notes are beamed within each beat group of the time signature (and in 4/4
    four plain quavers share a beam across a half bar); rests break beams."""
    if root.find(".//beam") is not None:
        return  # the file already has beaming; keep it
    for part in root.findall("part"):
        divisions, beats, beat_type = 1, None, None
        measures = part.findall("measure")
        for idx, m in enumerate(measures):
            for a in m.findall("attributes"):
                if a.findtext("divisions"):
                    divisions = int(a.findtext("divisions"))
                if a.find("time") is not None:
                    try:
                        beats = int(a.findtext("time/beats"))
                        beat_type = int(a.findtext("time/beat-type"))
                    except (TypeError, ValueError):
                        beats = beat_type = None
            if not beats or not beat_type:
                continue

            # (voice, onset, end, note) for each note that starts a chord, in quarters.
            events, pos = [], 0
            for c in m:
                if c.tag == "note":
                    if c.find("grace") is not None or c.find("chord") is not None:
                        continue
                    d = int(c.findtext("duration") or 0)
                    events.append((c.findtext("voice") or "1", Fraction(pos, divisions),
                                   Fraction(pos + d, divisions), c))
                    pos += d
                elif c.tag == "backup":
                    pos -= int(c.findtext("duration") or 0)
                elif c.tag == "forward":
                    pos += int(c.findtext("duration") or 0)
            if not events:
                continue
            bar = Fraction(beats * 4, beat_type)
            filled = max(end for _, _, end, _ in events)
            # In a pick-up bar the beats are counted from the end of the bar.
            shift = bar - filled if idx == 0 and filled < bar else Fraction(0)
            group = beam_group_length(beats, beat_type)

            for voice in sorted({v for v, *_ in events}):
                notes = sorted((e for e in events if e[0] == voice), key=lambda e: e[1])

                def beamable(note: ET.Element) -> bool:
                    return note.find("rest") is None and note.findtext("type") in BEAM_COUNT

                def runs_in(span: Fraction) -> list[list[ET.Element]]:
                    """Unbroken runs of beamable notes that stay inside one span."""
                    runs, current, last_key, last_end = [], [], None, None
                    for _, onset, end, note in notes:
                        key = (onset + shift) // span
                        fits = (end + shift) <= (key + 1) * span
                        if beamable(note) and fits and key == last_key and onset == last_end:
                            current.append(note)
                        else:
                            if len(current) > 1:
                                runs.append(current)
                            current = [note] if beamable(note) and fits else []
                        last_key, last_end = key, end
                    if len(current) > 1:
                        runs.append(current)
                    return runs

                runs = runs_in(group)
                if (beats, beat_type) == (4, 4):  # four plain quavers per half bar
                    halves = [r for r in runs_in(Fraction(2)) if len(r) == 4
                              and all(n.findtext("type") == "eighth" and n.find("dot") is None
                                      for n in r)]
                    joined = {id(n) for r in halves for n in r}
                    runs = halves + [r for r in runs if not joined & {id(n) for n in r}]
                for run in runs:
                    _beam_run(run)


# General MIDI programs (1-based) for instrument names the user types, checked in order
# (so "alto sax" wins over "alto", "bass clarinet" over "clarinet").
GM_PROGRAMS = [
    ("piccolo", 73), ("flute", 74), ("oboe", 69), ("cor anglais", 70), ("english horn", 70),
    ("clarinet", 72), ("bassoon", 71), ("recorder", 75),
    ("soprano sax", 65), ("alto sax", 66), ("tenor sax", 67), ("baritone sax", 68),
    ("saxophone", 66), ("horn", 61), ("trumpet", 57), ("cornet", 57), ("trombone", 58),
    ("tuba", 59), ("timpani", 48), ("xylophone", 14), ("marimba", 13), ("vibraphone", 12),
    ("glockenspiel", 10), ("celesta", 9), ("harpsichord", 7), ("organ", 20),
    ("accordion", 22), ("harmonica", 23), ("piano", 1), ("keyboard", 1), ("harp", 47),
    ("bass guitar", 34), ("electric guitar", 27), ("guitar", 25),
    ("violoncello", 43), ("cello", 43), ("viola", 42), ("violin", 41), ("fiddle", 41),
    ("double bass", 44), ("contrabass", 44), ("string bass", 44),
    ("choir", 53), ("soprano", 54), ("mezzo", 54), ("alto", 54), ("tenor", 54),
    ("baritone", 54), ("voice", 54), ("vocal", 54),
]


def gm_program(name: str) -> int | None:
    lowered = name.lower()
    return next((program for key, program in GM_PROGRAMS if key in lowered), None)


def set_instruments(root: ET.Element, names: list[str] | None) -> list[str]:
    """Name the parts, top to bottom, from `names`. Single-staff parts without a name
    are left undefined (no name, instrument or sound): homr calls them all "Voice",
    which made Sibelius give e.g. a cello line a singer's range and colour its notes
    red. Braced two-staff parts keep homr's "Piano" unless named.
    Returns the part names as written ("" = undefined)."""
    names = [n.strip() for n in names or []]
    part_list = root.find("part-list")
    written = []
    for i, sp in enumerate(part_list.findall("score-part") if part_list is not None else []):
        pid = sp.get("id")
        part = root.find(f"part[@id='{pid}']")
        two_staves = part is not None and part.findtext("measure/attributes/staves") == "2"
        name = names[i] if i < len(names) else ""
        if not name and two_staves:
            written.append(sp.findtext("part-name") or "")
            continue
        # Drop homr's instrument definition, and any notes that point at it.
        for tag in ("score-instrument", "midi-device", "midi-instrument"):
            for el in sp.findall(tag):
                sp.remove(el)
        for note in part.iter("note") if part is not None else ():
            for ref in note.findall("instrument"):
                note.remove(ref)
        part_name = sp.find("part-name")
        if part_name is None:
            part_name = ET.SubElement(sp, "part-name")
        part_name.text = name
        if name:
            instrument_id = f"{pid}-I1"
            score_instrument = ET.SubElement(sp, "score-instrument", id=instrument_id)
            ET.SubElement(score_instrument, "instrument-name").text = name
            program = gm_program(name)
            if program:
                channel = [c for c in range(1, 17) if c != 10][i % 15]  # 10 is drums
                midi = ET.SubElement(sp, "midi-instrument", id=instrument_id)
                ET.SubElement(midi, "midi-channel").text = str(channel)
                ET.SubElement(midi, "midi-program").text = str(program)
        written.append(name)
    return written


def clean_for_sibelius(root: ET.Element, title: str | None, composer: str | None,
                       instruments: list[str] | None = None) -> list[str]:
    """Rewrite homr's output for Sibelius 7. Returns the part names written."""
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

    parts = set_instruments(root, instruments)
    downgrade_to_2_0(root)

    for measure in root.iter("measure"):
        merge_attributes(measure)

    for note in root.iter("note"):
        if note.find("grace") is not None:  # grace notes may not have a duration
            for d in note.findall("duration"):
                note.remove(d)

    normalise_durations(root)
    add_beams(root)

    for el in root.iter():
        order = CHILD_ORDER.get(el.tag)
        if order:
            reorder_children(el, order)
    return parts


def check_measures(root: ET.Element) -> list[str]:
    """Return measure numbers whose contents don't add up to the time signature
    (in any part; each number listed once)."""
    bad: list[str] = []
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
            number = m.get("number", str(idx + 1))
            if (length > expected or (length < expected and not is_edge)) and number not in bad:
                bad.append(number)
    return sorted(bad, key=lambda n: (len(n), n))


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
    parts: list[str] = field(default_factory=list)  # top to bottom; "" = undefined
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
            holder: ProcessHolder | None = None,
            instruments: list[str] | None = None) -> Result:
    """Recognise the pages (joined in order into one score) and write Sibelius-ready
    MusicXML. `instruments` names the parts top to bottom; unnamed ones stay undefined."""
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

    parts = clean_for_sibelius(root, title, composer, instruments)
    given = [n.strip() for n in instruments or []]
    while given and not given[-1]:
        given.pop()
    if given and len(given) != len(parts):
        warnings.append(f"You gave {len(given)} instrument name(s) but the score has "
                        f"{len(parts)} part(s); they were applied top to bottom.")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_xml(root, out)
    return Result(output=out,
                  bars=len(root.findall("part")[0].findall("measure")),
                  notes=sum(1 for n in root.iter("note") if n.find("rest") is None),
                  bad_bars=check_measures(root),
                  seconds=time.time() - started,
                  device=device,
                  parts=parts,
                  warnings=warnings)


def describe_parts(parts: list[str]) -> str:
    """e.g. "3 parts: Violin, Violin, (undefined)"."""
    shown = ", ".join(p or "(undefined)" for p in parts)
    return f"{len(parts)} part{'s' if len(parts) != 1 else ''}: {shown}"


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
    ap.add_argument("--instruments", metavar="LIST",
                    help='instrument per part, top to bottom, e.g. "Violin, Violin, Cello" '
                         "(default: leave single-staff parts undefined)")
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
                    log=print if args.verbose else quiet,
                    instruments=args.instruments.split(",") if args.instruments else None)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e))

    print(f"Done in {r.seconds:.0f}s: {r.bars} bars, {r.notes} notes, {describe_parts(r.parts)}.")
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
