"""Window for score2xml: pick pages, choose CPU / NVIDIA / AMD, convert, open in Sibelius.

Start it with "Score to MusicXML.bat" (or: .venv\\Scripts\\pythonw score2xml_gui.py [files]).
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import score2xml as core

try:  # drag-and-drop onto the window; optional
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    TkinterDnD = None

APP_NAME = "Score to MusicXML"
SETTINGS_FILE = Path(os.environ.get("APPDATA") or Path.home()) / "score2xml" / "settings.json"
FILETYPES = [("Scores (images or PDF)", " ".join(f"*{e}" for e in sorted(core.INPUT_EXTS))),
             ("All files", "*.*")]
MAX_LOG_LINES = 3000
PROCESSORS = ["auto", "cpu", "nvidia", "amd"]


# --------------------------------------------------------------------------- helpers

def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> None:
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError:
        pass


def find_sibelius(saved: str | None) -> Path | None:
    """Sibelius.exe from the settings, the registry, or the usual install folders."""
    candidates: list[Path] = [Path(saved)] if saved else []
    try:
        import winreg

        key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Sibelius.exe"
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(hive, key) as k:
                    candidates.append(Path(winreg.QueryValue(k, None).strip('"')))
            except OSError:
                pass
    except ImportError:
        pass
    roots = {os.environ.get(v) for v in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432")}
    for root in filter(None, roots):
        for pattern in ("Avid/Sibelius*/Sibelius.exe", "Sibelius Software/Sibelius*/Sibelius.exe"):
            candidates += sorted(Path(root).glob(pattern), reverse=True)
    return next((c for c in candidates if c.is_file()), None)


def show_in_folder(path: Path) -> None:
    if os.name == "nt":
        subprocess.Popen(f'explorer /select,"{path}"')
    else:
        subprocess.Popen(["xdg-open", str(path.parent)])


def expand_inputs(paths: list[str]) -> tuple[list[Path], list[str]]:
    """Supported files from a list of files/folders, plus names that were skipped."""
    found, skipped = [], []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found += sorted(f for f in p.iterdir() if f.suffix.lower() in core.INPUT_EXTS)
        elif p.suffix.lower() in core.INPUT_EXTS and p.is_file():
            found.append(p)
        else:
            skipped.append(p.name)
    return found, skipped


# --------------------------------------------------------------------------- app

class App:
    def __init__(self, root: tk.Tk, initial_files: list[str]) -> None:
        self.root = root
        self.settings = load_settings()
        self.q: queue.Queue = queue.Queue()
        self.pages: list[Path] = []
        self.gpus: list[str] | None = None  # None = still detecting
        self.holder: core.ProcessHolder | None = None
        self.busy = False
        self.job_started = 0.0
        self.job_label = ""
        self.page_seen: list[str] = []
        self.output_is_auto = True
        self.output_confirmed = False
        self._setting_output = False
        self.last_result: core.Result | None = None

        root.title(APP_NAME)
        root.minsize(660, 640)
        self._style()
        self._build()

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self.poll)
        threading.Thread(target=lambda: self.q.put(("gpus", core.detect_gpus())),
                         daemon=True).start()
        self.refresh_processors()
        if initial_files:
            self.add_paths(initial_files)

    # ------------------------------------------------------------------ layout

    def _style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        base = ("Segoe UI", 10)
        self.root.option_add("*Listbox.font", base)  # "*Font" would override ttk styles
        style.configure(".", font=base)
        style.configure("Heading.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Hint.TLabel", foreground="#5f6368")
        style.configure("Good.TLabel", foreground="#1e7e34")
        style.configure("Warn.TLabel", foreground="#b45309")
        style.configure("Bad.TLabel", foreground="#b91c1c")
        style.configure("Big.TButton", font=("Segoe UI Semibold", 11), padding=(18, 6))

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        self.controls: list[tk.Widget] = []

        # Pages
        ttk.Label(outer, text="Pages", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        pages = ttk.Frame(outer)
        pages.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        pages.columnconfigure(0, weight=1)
        pages.rowconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        self.listbox = tk.Listbox(pages, height=6, selectmode="extended", activestyle="none",
                                  borderwidth=1, relief="solid", highlightthickness=0)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(pages, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.bind("<Delete>", lambda e: self.remove_pages())
        buttons = ttk.Frame(pages)
        buttons.grid(row=0, column=2, sticky="n", padx=(8, 0))
        for text, cmd in (("Add files…", self.browse_pages), ("Remove", self.remove_pages),
                          ("Move up", lambda: self.move_pages(-1)),
                          ("Move down", lambda: self.move_pages(1)), ("Clear", self.clear_pages)):
            b = ttk.Button(buttons, text=text, command=cmd, width=12)
            b.pack(fill="x", pady=(0, 4))
            self.controls.append(b)
        hint = ("Drop images or PDFs here, or use Add files. "
                if TkinterDnD else "Use Add files to pick images or PDFs. ")
        ttk.Label(outer, text=hint + "Pages are joined in this order into one score.",
                  style="Hint.TLabel").grid(row=2, column=0, sticky="w", pady=(4, 0))
        if TkinterDnD:
            for w in (self.listbox, self.root):
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self.on_drop)

        # Details
        form = ttk.Frame(outer)
        form.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        form.columnconfigure(1, weight=1)
        self.title_var, self.composer_var, self.output_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        ttk.Label(form, text="Title").grid(row=0, column=0, sticky="w", padx=(0, 10))
        e = ttk.Entry(form, textvariable=self.title_var)
        e.grid(row=0, column=1, sticky="ew", pady=2)
        self.controls.append(e)
        ttk.Label(form, text="Leave blank to use the title found on the page",
                  style="Hint.TLabel").grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(form, text="Composer").grid(row=1, column=0, sticky="w", padx=(0, 10))
        e = ttk.Entry(form, textvariable=self.composer_var)
        e.grid(row=1, column=1, sticky="ew", pady=2)
        self.controls.append(e)
        self.instruments_var = tk.StringVar()
        ttk.Label(form, text="Instruments").grid(row=2, column=0, sticky="w", padx=(0, 10))
        e = ttk.Entry(form, textvariable=self.instruments_var)
        e.grid(row=2, column=1, sticky="ew", pady=2)
        self.controls.append(e)
        ttk.Label(form, text="Top to bottom, e.g. Violin, Violin, Cello. Blank = undefined",
                  style="Hint.TLabel").grid(row=2, column=2, sticky="w", padx=(8, 0))
        ttk.Label(form, text="Save as").grid(row=3, column=0, sticky="w", padx=(0, 10))
        e = ttk.Entry(form, textvariable=self.output_var)
        e.grid(row=3, column=1, sticky="ew", pady=2)
        self.controls.append(e)
        b = ttk.Button(form, text="Browse…", command=self.browse_output)
        b.grid(row=3, column=2, sticky="w", padx=(8, 0))
        self.controls.append(b)
        self.output_var.trace_add("write", self._output_edited)

        # Processor
        ttk.Label(outer, text="Processor", style="Heading.TLabel").grid(
            row=4, column=0, sticky="w", pady=(14, 0))
        proc = ttk.Frame(outer)
        proc.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        proc.columnconfigure(1, weight=1)
        saved = self.settings.get("device", "auto")
        self.device_var = tk.StringVar(value=saved if saved in PROCESSORS else "auto")
        labels = {"auto": "Automatic (fastest available)",
                  **{k: b.label for k, b in core.BACKENDS.items()}}
        self.proc_rows: dict[str, tuple[ttk.Radiobutton, ttk.Label, ttk.Button | None]] = {}
        for i, key in enumerate(PROCESSORS):
            rb = ttk.Radiobutton(proc, text=labels[key], value=key, variable=self.device_var,
                                 command=self.refresh_processors)
            rb.grid(row=i, column=0, sticky="w", pady=1)
            status = ttk.Label(proc, style="Hint.TLabel")
            status.grid(row=i, column=1, sticky="w", padx=(12, 0))
            install = None
            if key != "auto":
                install = ttk.Button(proc, text="Install", command=lambda k=key: self.install(k))
                install.grid(row=i, column=2, sticky="e")
            self.proc_rows[key] = (rb, status, install)
        self.gpu_label = ttk.Label(proc, text="Graphics card: checking…", style="Hint.TLabel")
        self.gpu_label.grid(row=len(PROCESSORS), column=0, columnspan=3, sticky="w", pady=(4, 0))

        # Convert
        run = ttk.Frame(outer)
        run.grid(row=6, column=0, sticky="ew", pady=(16, 0))
        run.columnconfigure(2, weight=1)
        self.convert_btn = ttk.Button(run, text="Convert", style="Big.TButton",
                                      command=self.on_convert)
        self.convert_btn.grid(row=0, column=0, sticky="w")
        self.cancel_btn = ttk.Button(run, text="Cancel", command=self.on_cancel, state="disabled")
        self.cancel_btn.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.progress = ttk.Progressbar(run, mode="determinate", value=0)
        self.progress.grid(row=0, column=2, sticky="ew", padx=(16, 0))
        self.status = ttk.Label(outer, text="Add the pages of a score to get started.")
        self.status.grid(row=7, column=0, sticky="w", pady=(6, 0))

        # Result
        self.result_frame = ttk.Frame(outer)
        self.result_frame.grid(row=8, column=0, sticky="ew", pady=(10, 0))
        self.result_frame.columnconfigure(0, weight=1)
        self.result_text = ttk.Label(self.result_frame, wraplength=600, justify="left")
        self.result_text.grid(row=0, column=0, columnspan=3, sticky="w")
        self.result_note = ttk.Label(self.result_frame, wraplength=600, justify="left",
                                     style="Warn.TLabel")
        self.result_note.grid(row=1, column=0, columnspan=3, sticky="w")
        actions = ttk.Frame(self.result_frame)
        actions.grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.sibelius_btn = ttk.Button(actions, text="Open in Sibelius", command=self.open_in_sibelius)
        self.sibelius_btn.pack(side="left")
        ttk.Button(actions, text="Show in folder",
                   command=lambda: self.last_result and show_in_folder(self.last_result.output)
                   ).pack(side="left", padx=(8, 0))
        self.result_frame.grid_remove()

        # Log
        self.log_shown = tk.BooleanVar(value=False)
        ttk.Checkbutton(outer, text="Show details", variable=self.log_shown,
                        command=self.toggle_log).grid(row=9, column=0, sticky="w", pady=(10, 0))
        self.log_frame = ttk.Frame(outer)
        self.log_frame.grid(row=10, column=0, sticky="nsew", pady=(4, 0))
        self.log_frame.columnconfigure(0, weight=1)
        self.log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(self.log_frame, height=10, wrap="none", font=("Consolas", 9),
                           borderwidth=1, relief="solid", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        ys = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log.yview)
        ys.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=ys.set)
        self.log_frame.grid_remove()
        self.outer = outer

    def toggle_log(self) -> None:
        if self.log_shown.get():
            self.log_frame.grid()
            self.outer.rowconfigure(10, weight=2)
        else:
            self.log_frame.grid_remove()
            self.outer.rowconfigure(10, weight=0)

    # ------------------------------------------------------------------ pages

    def add_paths(self, paths: list[str]) -> None:
        found, skipped = expand_inputs(paths)
        known = {p.resolve() for p in self.pages}
        for p in found:
            if p.resolve() not in known:
                self.pages.append(p)
                known.add(p.resolve())
        if found:
            self.settings["last_dir"] = str(found[-1].parent)
        self.refresh_pages()
        if skipped:
            self.set_status(f"Skipped (not an image or PDF): {', '.join(skipped[:5])}")
        elif found:
            self.set_status("Ready to convert.")

    def on_drop(self, event) -> None:
        if not self.busy:
            self.add_paths(list(self.root.tk.splitlist(event.data)))

    def browse_pages(self) -> None:
        names = filedialog.askopenfilenames(title="Add score pages", filetypes=FILETYPES,
                                            initialdir=self.settings.get("last_dir"))
        if names:
            self.add_paths(list(names))

    def selected(self) -> list[int]:
        return list(self.listbox.curselection())

    def remove_pages(self) -> None:
        for i in reversed(self.selected()):
            del self.pages[i]
        self.refresh_pages()

    def move_pages(self, step: int) -> None:
        sel = self.selected()
        if not sel or (step < 0 and sel[0] == 0) or (step > 0 and sel[-1] == len(self.pages) - 1):
            return
        for i in (sel if step < 0 else reversed(sel)):
            self.pages[i + step], self.pages[i] = self.pages[i], self.pages[i + step]
        self.refresh_pages([i + step for i in sel])

    def clear_pages(self) -> None:
        self.pages.clear()
        self.refresh_pages()

    def refresh_pages(self, select: list[int] | None = None) -> None:
        self.listbox.delete(0, "end")
        for i, p in enumerate(self.pages, 1):
            self.listbox.insert("end", f" {i}.  {p.name}     —  {p.parent}")
        for i in select or []:
            self.listbox.selection_set(i)
        if self.output_is_auto:
            self.set_output(core.default_output(self.pages[0]) if self.pages else None)

    # ------------------------------------------------------------------ output path

    def set_output(self, path: Path | None, auto: bool = True, confirmed: bool = False) -> None:
        self._setting_output = True
        self.output_var.set(str(path) if path else "")
        self._setting_output = False
        self.output_is_auto, self.output_confirmed = auto, confirmed

    def _output_edited(self, *_):
        if not self._setting_output:
            self.output_is_auto, self.output_confirmed = False, False

    def browse_output(self) -> None:
        current = Path(self.output_var.get()) if self.output_var.get() else None
        name = filedialog.asksaveasfilename(
            title="Save MusicXML as", defaultextension=".xml",
            filetypes=[("MusicXML for Sibelius", "*.xml")],
            initialdir=str(current.parent) if current else self.settings.get("last_dir"),
            initialfile=current.name if current else "")
        if name:
            self.set_output(Path(name), auto=False, confirmed=True)

    # ------------------------------------------------------------------ processors

    def refresh_processors(self) -> None:
        gpus = self.gpus
        if gpus is not None:
            self.gpu_label.configure(text="Graphics card: " + (", ".join(gpus) or "none found"))
        for key in PROCESSORS:
            rb, status, install = self.proc_rows[key]
            if key == "auto":
                text = ("checking…" if gpus is None else
                        "uses " + core.BACKENDS[core.auto_device(gpus)].label)
                status.configure(text=text, style="Hint.TLabel")
                continue
            b = core.BACKENDS[key]
            ready = core.backend_ready(key)
            if key == "cpu":
                hardware, why_not = True, ""
            elif gpus is None:
                hardware, why_not = False, "checking…"
            elif key == "nvidia":
                hardware = core.gpu_vendor_present(gpus, "nvidia")
                why_not = "No NVIDIA graphics card found"
            else:  # DirectML runs on any DirectX 12 card, so it doesn't need an AMD one
                hardware, why_not = bool(gpus), "No graphics card found"

            if not hardware:
                text, style = why_not, "Hint.TLabel"
            elif ready:
                text, style = "Ready", "Good.TLabel"
                if key == "amd" and not core.gpu_vendor_present(gpus, "amd"):
                    text = f"Ready — no AMD card here, so it would run on your {gpus[0]}"
            else:
                text, style = f"Not installed (needs {b.size})", "Hint.TLabel"
            status.configure(text=text, style=style)
            rb.configure(state="normal" if ready and hardware and not self.busy else "disabled")
            if install:
                install.configure(state="disabled" if self.busy else "normal")
                if hardware and not ready:
                    install.grid()
                else:
                    install.grid_remove()
            if self.device_var.get() == key and not (ready and hardware) and gpus is not None:
                self.device_var.set("auto")
        self.proc_rows["auto"][0].configure(state="disabled" if self.busy else "normal")

    def resolved_device(self) -> str:
        choice = self.device_var.get()
        return core.auto_device(self.gpus or []) if choice == "auto" else choice

    # ------------------------------------------------------------------ jobs

    def set_busy(self, busy: bool, label: str = "") -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for w in self.controls:
            w.configure(state=state)
        self.convert_btn.configure(state=state)
        self.cancel_btn.configure(state="normal" if busy else "disabled")
        if busy:
            self.job_started, self.job_label, self.page_seen = time.time(), label, []
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
            self.set_status(label)
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate", value=0)
            self.holder = None
        self.refresh_processors()

    def start_job(self, label: str, work) -> None:
        self.holder = core.ProcessHolder()
        holder = self.holder
        self.set_busy(True, label)
        self.clear_log()

        def run():
            try:
                self.q.put(("done", work(holder)))
            except core.Cancelled:
                self.q.put(("cancelled", None))
            except Exception as e:  # shown to the user
                self.q.put(("error", str(e) or e.__class__.__name__))

        threading.Thread(target=run, daemon=True).start()

    def on_convert(self) -> None:
        if self.busy:
            return
        if not self.pages:
            messagebox.showinfo(APP_NAME, "Add at least one image or PDF first.")
            return
        missing = [p.name for p in self.pages if not p.is_file()]
        if missing:
            messagebox.showerror(APP_NAME, "These files can't be found any more:\n"
                                 + "\n".join(missing))
            return
        device = self.resolved_device()
        if not core.backend_ready(device):
            messagebox.showinfo(APP_NAME, f"{core.BACKENDS[device].label} isn't installed yet. "
                                "Click its Install button first.")
            return
        if self.output_is_auto or not self.output_var.get().strip():
            self.set_output(core.default_output(self.pages[0]))
        out = Path(self.output_var.get().strip()).with_suffix(".xml")
        if out.exists() and not self.output_is_auto and not self.output_confirmed:
            if not messagebox.askyesno(APP_NAME, f"{out.name} already exists. Replace it?"):
                return

        title = self.title_var.get().strip() or None
        composer = self.composer_var.get().strip() or None
        instruments = self.instruments_var.get().split(",") if self.instruments_var.get().strip() else None
        pages = list(self.pages)
        self.settings["device"] = self.device_var.get()
        save_settings(self.settings)
        self.result_frame.grid_remove()
        self.start_job(
            f"Reading {len(pages)} file(s) on {core.BACKENDS[device].label}…",
            lambda holder: core.convert(pages, out, title, composer, device,
                                        log=lambda line: self.q.put(("log", line)),
                                        holder=holder, instruments=instruments))

    def install(self, key: str) -> None:
        b = core.BACKENDS[key]
        if not messagebox.askyesno(
                APP_NAME, f"Install {b.label} support?\n\nThis needs {b.size} of disk space and "
                "takes a few minutes. You can keep the window open while it works."):
            return
        self.log_shown.set(True)
        self.toggle_log()

        def work(holder: core.ProcessHolder) -> str:
            core.install_backend(key, lambda line: self.q.put(("log", line)), holder)
            return key

        self.start_job(f"Installing {b.label} support…", work)

    def on_cancel(self) -> None:
        if self.holder:
            self.holder.cancel()
            self.set_status("Stopping…")

    def on_close(self) -> None:
        if self.busy:
            if not messagebox.askyesno(APP_NAME, "Stop the current job and close?"):
                return
            if self.holder:
                self.holder.cancel()
        save_settings(self.settings)
        self.root.destroy()

    # ------------------------------------------------------------------ results

    def show_result(self, r: core.Result) -> None:
        self.last_result = r
        self.result_text.configure(
            text=f"Saved {r.output.name} — {r.bars} bars, {r.notes} notes, "
                 f"{r.seconds:.0f} s on {core.BACKENDS[r.device].label}.\n"
                 f"{core.describe_parts(r.parts)}\n{r.output.parent}")
        notes = list(r.warnings)
        if r.bad_bars:
            shown = ", ".join(r.bad_bars[:20]) + (" …" if len(r.bad_bars) > 20 else "")
            notes.append(f"Check these bars in Sibelius — their rhythms don't add up: {shown}")
        self.result_note.configure(text="\n".join(notes))
        sib = find_sibelius(self.settings.get("sibelius"))
        self.sibelius_btn.configure(text="Open in Sibelius" if sib else "Open in Sibelius…")
        self.result_frame.grid()

    def open_in_sibelius(self) -> None:
        if not self.last_result:
            return
        sib = find_sibelius(self.settings.get("sibelius"))
        if not sib:
            name = filedialog.askopenfilename(
                title="Where is Sibelius.exe?", filetypes=[("Sibelius", "Sibelius.exe"),
                                                           ("Programs", "*.exe")],
                initialdir=os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles"))
            if not name:
                return
            sib = Path(name)
            self.settings["sibelius"] = str(sib)
            save_settings(self.settings)
            self.sibelius_btn.configure(text="Open in Sibelius")
        try:
            subprocess.Popen([str(sib), str(self.last_result.output)])
        except OSError as e:
            messagebox.showerror(APP_NAME, f"Couldn't start Sibelius:\n{e}")

    # ------------------------------------------------------------------ messages

    def set_status(self, text: str, style: str = "TLabel") -> None:
        self.status.configure(text=text, style=style)

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def append_log(self, lines: list[str]) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", "".join(ln[:400] + "\n" for ln in lines))
        excess = int(self.log.index("end-1c").split(".")[0]) - MAX_LOG_LINES
        if excess > 0:
            self.log.delete("1.0", f"{excess + 1}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def progress_from(self, line: str) -> None:
        """Turn homr / installer output into a short status line."""
        step = None
        if line.startswith("== "):
            step = line[3:]
        elif m := re.match(r"Processing (page\d+(?:_\d+)?)\.png", line):
            if m.group(1) not in self.page_seen:
                self.page_seen.append(m.group(1))
            step = f"Reading page {len(self.page_seen)}: finding the staves"
        elif m := re.match(r"Running TrOmr inference on staff image (\d+)", line):
            step = f"Reading page {max(len(self.page_seen), 1)}: staff {int(m.group(1)) + 1}"
        elif line.startswith("Downloading") and "model" in line:
            step = "Downloading recognition models (first run only)"
        elif line.startswith("Writing XML"):
            step = "Writing MusicXML"
        if step:
            self.job_label = step

    def poll(self) -> None:
        lines = []
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "log":
                    lines.append(data)
                    self.progress_from(data)
                elif kind == "gpus":
                    self.gpus = data
                    self.refresh_processors()
                elif kind == "done":
                    if lines:
                        self.append_log(lines)
                        lines = []
                    self.set_busy(False)
                    if isinstance(data, core.Result):
                        self.progress.configure(value=100)
                        self.set_status("Done.", "Good.TLabel")
                        self.show_result(data)
                    else:
                        self.set_status(f"{core.BACKENDS[data].label} is ready.", "Good.TLabel")
                elif kind == "cancelled":
                    self.set_busy(False)
                    self.set_status("Stopped.", "Warn.TLabel")
                elif kind == "error":
                    if lines:
                        self.append_log(lines)
                        lines = []
                    self.set_busy(False)
                    self.set_status("Something went wrong — see the message.", "Bad.TLabel")
                    messagebox.showerror(APP_NAME, data)
        except queue.Empty:
            pass
        if lines:
            self.append_log(lines)
        if self.busy:
            self.set_status(f"{self.job_label}   ({time.time() - self.job_started:.0f} s)")
        self.root.after(100, self.poll)


def main() -> None:
    if os.name == "nt":
        try:  # crisp text on high-DPI screens
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = TkinterDnD.Tk() if TkinterDnD else tk.Tk()
    App(root, sys.argv[1:])
    root.mainloop()


if __name__ == "__main__":
    main()
