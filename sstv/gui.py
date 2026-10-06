"""The SSTV Studio desktop window.

A single window with a transmit side and a receive side, both working on one
image at a time -- SSTV is a simplex mode, so the program never tries to send and
receive at once.

Long operations (encoding a multi-minute mode, playing it out, recording,
decoding) run on worker threads and report progress through callbacks that are
marshalled back to the Tk event loop, so the window never freezes and the
progress bar always moves.
"""

from __future__ import annotations

import os
import ctypes
import os
import queue
import sys
import threading
import time
import traceback
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import audio as audio_module
from . import decoder as decoder_module
from . import encoder as encoder_module
from . import modes as modes_module
from . import streaming as streaming_module
from . import version as version_module
from . import waterfall as waterfall_module

__all__ = ["SstvApp", "main", "run"]

APP_NAME = "SSTV Studio"
# Read from the single definition in sstv/version.py, so the title bar, --version,
# the documentation and the package name can never disagree.
APP_VERSION = version_module.VERSION

# Waterfall frequency spans, in Hz.  The first is the whole SSTV band with
# margin; the narrower ones zoom in so the sync pulse and the black/white limits
# can be read off precisely, which is how an operator checks tuning.
WATERFALL_SPANS = (
    "1000 - 2500 Hz",
    "1100 - 2400 Hz",
    "1300 - 2100 Hz",
    "1400 - 2000 Hz",
)
WATERFALL_WINDOWS = ("3 seconds", "6 seconds", "12 seconds", "30 seconds")
# History worth showing for each mode: enough lines to see the pattern repeat,
# but not so many that a fast mode's lines smear together.
WATERFALL_WINDOW_BY_MODE = {
    "Robot 36": "3 seconds",
    "Robot 72": "6 seconds",
    "Martin M1": "6 seconds",
    "Martin M2": "3 seconds",
    "Scottie S1": "6 seconds",
    "Scottie S2": "3 seconds",
    "Scottie DX": "12 seconds",
    "PD 50": "3 seconds",
    "PD 90": "6 seconds",
    "PD 120": "6 seconds",
    "PD 160": "6 seconds",
    "PD 180": "6 seconds",
    "PD 240": "12 seconds",
    "PD 290": "12 seconds",
}
# Live repaint interval.  The transform itself is well under a millisecond; this
# is paced so drawing cannot monopolise the event loop while a recording runs.
WATERFALL_INTERVAL_MS = 66

_PREVIEW_BG = "#1b1d21"
_PREVIEW_FG = "#3a3f47"
_TEXT = "#e8eaed"
_MUTED = "#9aa0a6"
_ACCENT = "#4c8bf5"
_OK = "#34a853"
_WARN = "#f9ab00"
_ERROR = "#ea4335"


@dataclass
class _Task:
    """A unit of work for the worker thread."""

    fn: Callable[[], None]
    name: str = ""


class SstvApp:
    """The application window and all of its state."""

    def __init__(self, root=None, sample_rate: int = audio_module.DEFAULT_SAMPLE_RATE) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.sample_rate = sample_rate
        self.owns_root = root is None
        self.root = root if root is not None else tk.Tk()
        self.root.title(f"{APP_NAME} {APP_VERSION}")
        self.root.configure(bg=_PREVIEW_BG)

        # -- state --------------------------------------------------------
        self.tx_image = None                 # PIL image being sent
        self.tx_audio: np.ndarray | None = None
        self.tx_encoding = False
        self.player = audio_module.AudioPlayer(sample_rate)
        self.recorder = audio_module.AudioRecorder(sample_rate)
        self.rx_image: np.ndarray | None = None
        self.rx_result: decoder_module.DecodeResult | None = None
        self._tasks: queue.Queue[_Task] = queue.Queue()
        self._events: queue.Queue[tuple] = queue.Queue()
        self._worker_busy = False
        self._closing = False
        self._resize_job = None

        # -- waterfall ----------------------------------------------------
        # Only ever touched from the Tk thread: the recorder thread appends to
        # its own buffer under a lock, and this side pulls the new bytes out.
        self.waterfall = waterfall_module.Waterfall(
            sample_rate, seconds=self._waterfall_seconds())
        self._waterfall_photo = None
        self._waterfall_cursor = 0
        self._waterfall_dirty = True
        self._waterfall_colormap = waterfall_module.build_colormap()
        self._waterfall_job = None

        # -- progressive decoding -----------------------------------------
        # A transmission takes up to five minutes; decoding only once it has
        # finished means an empty panel for all of it.  This decoder consumes the
        # same audio the waterfall does and fills the picture in line by line.
        self._live_decoder = None
        self._live_finished = False
        self._live_photo = None
        self._live_notes: list[str] = []
        self._live_progress = None

        self._build_style()
        self._build_widgets()
        # Sized once the state it uses exists (_closing, _resize_job), because
        # applying the size binds a resize handler.
        self._apply_window_size()
        self._refresh_devices()
        self._start_worker()
        self._schedule(80, self._pump_events)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------
    # window sizing
    # ------------------------------------------------------------------
    def _monitor_work_area(self) -> tuple[int, int, int, int]:
        """Work area of the monitor the window sits on, as ``(x, y, w, h)``.

        This is what stops the window from spanning two screens.  Tk's virtual
        root covers *all* monitors glued together -- 4480x1262 on a pair of
        2240x1262 displays, for instance -- so centring a 1220-pixel window on it
        invites Windows to spread the window across both monitors while it looks
        for somewhere sensible to put it.  Asking the window manager which
        monitor the window is on gives the right rectangle to size and centre
        against.

        Returns ``(0, 0, 0, 0)`` when the query is unavailable, and the caller
        then falls back to Tk's own idea of the desktop.
        """
        if sys.platform != "win32":
            return (0, 0, 0, 0)

        class _RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        class _MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", _RECT),
                        ("rcWork", _RECT), ("dwFlags", ctypes.c_ulong)]

        try:
            user32 = ctypes.windll.user32
            self.root.update_idletasks()
            hwnd = int(self.root.winfo_id())
            monitor = user32.MonitorFromWindow(wintypes.HWND(hwnd), 2)  # NEAREST
            info = _MONITORINFO()
            info.cbSize = ctypes.sizeof(_MONITORINFO)
            if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                return (0, 0, 0, 0)
            work = info.rcWork
            return (work.left, work.top,
                    work.right - work.left, work.bottom - work.top)
        except Exception:
            return (0, 0, 0, 0)

    def _screen_work_area(self) -> tuple[int, int, int, int]:
        """Available desktop area as ``(x, y, width, height)``.

        Uses the *work area* rather than the screen size so the window never ends
        up underneath the taskbar, and takes the origin into account because a
        monitor placed to the left of or above the primary one has a negative one.
        """
        area = self._monitor_work_area()
        if area[2] > 200 and area[3] > 200:
            return area
        root = self.root
        try:
            if root.winfo_vrootwidth() > 1 and root.winfo_vrootheight() > 1:
                return (root.winfo_vrootx(), root.winfo_vrooty(),
                        root.winfo_vrootwidth(), root.winfo_vrootheight())
        except Exception:
            pass
        return (0, 0, root.winfo_screenwidth(), root.winfo_screenheight())

    def _apply_window_size(self) -> None:
        """Size the window to the actual screen instead of assuming one.

        Hard-coding a default and a minimum is what made the program unusable on
        a smaller display: a window taller than the desktop overflows it, and
        because the controls are laid out from the top, the ones at the bottom --
        the waterfall settings and the Save/Clear buttons -- end up off the
        bottom edge where they cannot be reached at all.
        """
        x, y, area_w, area_h = self._screen_work_area()
        margin = 30
        # Never ask for more room than the desktop has, leaving a margin so the
        # title bar and borders stay on screen.
        max_w = max(560, area_w - margin)
        max_h = max(420, area_h - margin)
        want_w, want_h = 1220, 880
        width = min(want_w, max_w)
        height = min(want_h, max_h)
        self.root.geometry(f"{width}x{height}+{x + max(0, (area_w - width) // 2)}"
                           f"+{y + max(0, (area_h - height) // 2)}")
        # A minimum larger than the desktop would make the window impossible to
        # fit; on a short screen the layout adapts (the sidebar scrolls) instead.
        self.root.minsize(min(860, max_w), min(520, max_h))
        if height < want_h or width < want_w:
            # Remember the squeeze so the layout can start compact.
            self._started_compact = True
        else:
            self._started_compact = False

        # Dragging the window to a monitor with a different size must not leave a
        # minimum size that is larger than that screen, or the window could not be
        # made small enough to reach the controls at the bottom.  Re-check
        # whenever the window settles after a resize or a move.
        self.root.bind("<Configure>", self._on_window_configure, add="+")

    def _on_window_configure(self, event=None) -> None:
        if event is not None and getattr(event, "widget", None) is not self.root:
            return
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self._schedule(220, self._refresh_minimum_size)

    def _refresh_minimum_size(self) -> None:
        self._resize_job = None
        _x, _y, area_w, area_h = self._screen_work_area()
        max_w = max(560, area_w - 30)
        max_h = max(420, area_h - 30)
        try:
            self.root.minsize(min(860, max_w), min(520, max_h))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_style(self) -> None:
        ttk = self.ttk
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=_PREVIEW_BG, foreground=_TEXT)
        style.configure("TFrame", background=_PREVIEW_BG)
        style.configure("TLabel", background=_PREVIEW_BG, foreground=_TEXT)
        style.configure("Muted.TLabel", background=_PREVIEW_BG, foreground=_MUTED)
        style.configure("Head.TLabel", background=_PREVIEW_BG, foreground=_TEXT,
                        font=("Segoe UI", 11, "bold"))
        style.configure("Title.TLabel", background=_PREVIEW_BG, foreground=_TEXT,
                        font=("Segoe UI", 15, "bold"))
        style.configure("TButton", padding=(10, 6))
        style.configure("Accent.TButton", padding=(12, 7))
        style.configure("TNotebook", background=_PREVIEW_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 8))
        style.configure("TLabelframe", background=_PREVIEW_BG, foreground=_TEXT)
        style.configure("TLabelframe.Label", background=_PREVIEW_BG, foreground=_MUTED)

    def _build_widgets(self) -> None:
        tk, ttk = self.tk, self.ttk
        root = self.root

        header = ttk.Frame(root, padding=(14, 12, 14, 4))
        header.pack(fill="x")
        ttk.Label(header, text=APP_NAME, style="Title.TLabel").pack(side="left")
        ttk.Label(header, text=f"  {APP_VERSION}", style="Muted.TLabel").pack(side="left", pady=(6, 0))
        self.status = ttk.Label(header, text="Ready", style="Muted.TLabel")
        self.status.pack(side="right", pady=(6, 0))

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(4, 6))

        self.tx_tab = ttk.Frame(self.notebook, padding=10)
        self.rx_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tx_tab, text="  Transmit  ")
        self.notebook.add(self.rx_tab, text="  Receive  ")

        self._build_tx_tab()
        self._build_rx_tab()

        self.progress = ttk.Progressbar(root, mode="determinate", maximum=1000)
        self.progress.pack(fill="x", padx=12, pady=(0, 10))

    # -- transmit ------------------------------------------------------
    def _build_tx_tab(self) -> None:
        tk, ttk = self.tk, self.ttk
        tab = self.tx_tab

        left = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True)
        right_holder = ttk.Frame(tab, width=250)
        right_holder.pack(side="right", fill="y", padx=(12, 0))
        right_holder.pack_propagate(False)
        right = self._make_scrollable(right_holder)

        ttk.Label(left, text="Image to send", style="Head.TLabel").pack(anchor="w")
        # Wrapped in a sized container for the same reason as the receive
        # preview: a Label holding an image reports the image's size as its own
        # request and then takes over the column, squeezing everything below it.
        tx_holder = tk.Frame(left, background=_PREVIEW_BG, width=460, height=300)
        tx_holder.pack(fill="both", expand=True, pady=(6, 0))
        tx_holder.pack_propagate(False)
        self.tx_preview = tk.Label(tx_holder, background=_PREVIEW_BG,
                                   foreground=_MUTED, text="No image loaded")
        self.tx_preview.pack(fill="both", expand=True)
        self.tx_info = ttk.Label(left, text="", style="Muted.TLabel")
        self.tx_info.pack(anchor="w", pady=(6, 0))

        box = ttk.Labelframe(right, text="Source", padding=10)
        box.pack(fill="x")
        ttk.Button(box, text="Open image...", command=self.tx_open_image).pack(fill="x", pady=2)
        ttk.Button(box, text="Use test card", command=self.tx_test_card).pack(fill="x", pady=2)
        ttk.Button(box, text="From received image", command=self.tx_from_rx).pack(fill="x", pady=2)

        modebox = ttk.Labelframe(right, text="Mode", padding=10)
        modebox.pack(fill="x", pady=(10, 0))
        self.mode_var = tk.StringVar(value="Robot 36")
        self.mode_combo = ttk.Combobox(modebox, textvariable=self.mode_var,
                                       values=list(modes_module.names()), state="readonly")
        self.mode_combo.pack(fill="x")
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self._update_mode_info())
        self.mode_info = ttk.Label(modebox, text="", style="Muted.TLabel", wraplength=190,
                                   justify="left")
        self.mode_info.pack(anchor="w", pady=(6, 0))

        outbox = ttk.Labelframe(right, text="Send via", padding=10)
        outbox.pack(fill="x", pady=(10, 0))
        self.tx_device_var = tk.StringVar()
        self.tx_device_combo = ttk.Combobox(outbox, textvariable=self.tx_device_var, state="readonly")
        self.tx_device_combo.pack(fill="x")
        ttk.Button(outbox, text="\u25b6  Transmit", style="Accent.TButton",
                   command=self.tx_transmit).pack(fill="x", pady=(8, 2))
        self.tx_stop_btn = ttk.Button(outbox, text="\u25a0  Stop", command=self.tx_stop,
                                      state="disabled")
        self.tx_stop_btn.pack(fill="x", pady=2)
        ttk.Button(outbox, text="Save audio file...", command=self.tx_save_wav).pack(fill="x", pady=2)

        tick = ttk.Frame(outbox)
        tick.pack(fill="x", pady=(6, 0))
        self.tx_progress_label = ttk.Label(tick, text="Idle", style="Muted.TLabel")
        self.tx_progress_label.pack(anchor="w")

        self._update_mode_info()

    # -- receive -------------------------------------------------------
    def _build_rx_tab(self) -> None:
        tk, ttk = self.tk, self.ttk
        tab = self.rx_tab

        left = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True)
        right_holder = ttk.Frame(tab, width=250)
        right_holder.pack(side="right", fill="y", padx=(12, 0))
        right_holder.pack_propagate(False)
        right = self._make_scrollable(right_holder)

        # The picture and the waterfall share the column through a draggable
        # split.  Packing them as siblings does not work: a Label holding an
        # image reports its own requested size and the packer then hands it the
        # whole column, pushing the waterfall off the bottom of the window.  A
        # PanedWindow arbitrates between them and lets the operator rebalance the
        # two by hand, which is also what they will want when a mode is fast
        # enough that the waterfall deserves more room than the picture.
        split = tk.PanedWindow(left, orient="vertical", sashwidth=6, bd=0,
                               background=_PREVIEW_BG, sashrelief="flat")
        split.pack(fill="both", expand=True)

        picture = ttk.Frame(split)
        water = ttk.Frame(split)
        split.add(picture, stretch="always", minsize=150)
        split.add(water, stretch="always", minsize=110)

        ttk.Label(picture, text="Received image", style="Head.TLabel").pack(anchor="w")
        # The label is placed inside a container with an explicit geometry
        # request.  A bare Label that holds an image reports the image's own size
        # as its request, and turning propagation off on it does not reliably stop
        # that; a Frame with a width and height does honour the request, which is
        # what keeps the picture from swallowing the waterfall below it.
        preview_holder = tk.Frame(picture, background=_PREVIEW_BG,
                                  width=440, height=230)
        preview_holder.pack(fill="both", expand=True, pady=(6, 0))
        preview_holder.pack_propagate(False)
        self.rx_preview = tk.Label(preview_holder, background=_PREVIEW_BG,
                                   foreground=_MUTED, text="Nothing received yet")
        self.rx_preview.pack(fill="both", expand=True)

        ttk.Label(water, text="Waterfall", style="Head.TLabel").pack(anchor="w")
        peakrow = ttk.Frame(water)
        peakrow.pack(fill="x")
        self.rx_peak_label = ttk.Label(peakrow, text="", style="Muted.TLabel")
        self.rx_peak_label.pack(side="right")
        self.rx_waterfall = tk.Canvas(water, height=40, background="#0a0b0e",
                                      highlightthickness=1,
                                      highlightbackground=_PREVIEW_FG)
        self.rx_waterfall.pack(fill="both", expand=True, pady=(4, 0))
        self.rx_waterfall.bind("<Configure>", self._on_waterfall_resize)

        self.rx_info = ttk.Label(left, text="", style="Muted.TLabel", justify="left")
        self.rx_info.pack(anchor="w", pady=(8, 0))

        srcbox = ttk.Labelframe(right, text="Source", padding=10)
        srcbox.pack(fill="x")
        # Two ways in: a microphone / line input, or the computer's own output.
        # System sound is what makes it possible to decode a receiver that is
        # feeding the speakers, or a WebSDR playing in a browser, with no cable.
        self.rx_source_var = tk.StringVar(value="microphone")
        ttk.Radiobutton(srcbox, text="Microphone or line input", value="microphone",
                        variable=self.rx_source_var,
                        command=self._on_source_changed).pack(anchor="w")
        self.rx_loopback_radio = ttk.Radiobutton(
            srcbox, text="System sound (what is playing)", value="loopback",
            variable=self.rx_source_var, command=self._on_source_changed)
        self.rx_loopback_radio.pack(anchor="w", pady=(2, 6))
        devrow = ttk.Frame(srcbox)
        devrow.pack(fill="x", pady=(0, 6))
        ttk.Label(devrow, text="Input", style="Muted.TLabel").pack(anchor="w")
        self.rx_device_var = tk.StringVar()
        self.rx_device_combo = ttk.Combobox(srcbox, textvariable=self.rx_device_var, state="readonly")
        self.rx_device_combo.pack(fill="x")
        self.rx_record_btn = ttk.Button(srcbox, text="\u25cf  Record from microphone",
                                        style="Accent.TButton", command=self.rx_toggle_record)
        self.rx_record_btn.pack(fill="x", pady=(8, 2))
        ttk.Button(srcbox, text="Open audio file...", command=self.rx_open_file).pack(fill="x", pady=2)
        self.rx_loopback_hint = ttk.Label(
            srcbox, text="", style="Muted.TLabel", wraplength=210, justify="left")
        self.rx_loopback_hint.pack(anchor="w", pady=(4, 0))
        self.rx_level = ttk.Progressbar(srcbox, mode="determinate", maximum=100)
        self.rx_level.pack(fill="x", pady=(6, 0))
        self.rx_level_label = ttk.Label(srcbox, text="Level", style="Muted.TLabel")
        self.rx_level_label.pack(anchor="w")

        wfbox = ttk.Labelframe(right, text="Waterfall", padding=10)
        wfbox.pack(fill="x", pady=(10, 0))
        spanrow = ttk.Frame(wfbox)
        spanrow.pack(fill="x")
        ttk.Label(spanrow, text="Frequency span", style="Muted.TLabel").pack(anchor="w")
        self.wf_span_var = tk.StringVar(value=WATERFALL_SPANS[0])
        self.wf_span_combo = ttk.Combobox(spanrow, textvariable=self.wf_span_var,
                                          values=list(WATERFALL_SPANS), state="readonly")
        self.wf_span_combo.pack(fill="x")
        self.wf_span_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_waterfall_settings())
        timerow = ttk.Frame(wfbox)
        timerow.pack(fill="x", pady=(6, 0))
        ttk.Label(timerow, text="History", style="Muted.TLabel").pack(anchor="w")
        self.wf_seconds_var = tk.StringVar(value=WATERFALL_WINDOWS[1])
        self.wf_seconds_combo = ttk.Combobox(timerow, textvariable=self.wf_seconds_var,
                                             values=list(WATERFALL_WINDOWS), state="readonly")
        self.wf_seconds_combo.pack(fill="x")
        self.wf_seconds_combo.bind("<<ComboboxSelected>>",
                                  lambda e: self._apply_waterfall_settings())
        self.wf_scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(wfbox, text="Show frequency scale", variable=self.wf_scale_var,
                        command=self._mark_waterfall_dirty).pack(anchor="w", pady=(6, 0))
        ttk.Button(wfbox, text="Suggested history for mode",
                   command=self._suggest_waterfall_window).pack(fill="x", pady=(6, 0))
        ttk.Button(wfbox, text="Clear waterfall",
                   command=self._clear_waterfall).pack(fill="x", pady=2)

        optbox = ttk.Labelframe(right, text="Decoding", padding=10)
        optbox.pack(fill="x", pady=(10, 0))
        self.auto_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(optbox, text="Detect mode from VIS header",
                        variable=self.auto_mode_var,
                        command=self._toggle_auto_mode).pack(anchor="w")
        self.rx_mode_var = tk.StringVar(value="Robot 36")
        self.rx_mode_combo = ttk.Combobox(optbox, textvariable=self.rx_mode_var,
                                          values=list(modes_module.names()), state="disabled")
        self.rx_mode_combo.pack(fill="x", pady=(4, 0))
        self.save_rx_btn = ttk.Button(optbox, text="Save image...", command=self.rx_save_image,
                                      state="disabled")
        self.save_rx_btn.pack(fill="x", pady=(8, 0))
        ttk.Button(optbox, text="Clear", command=self.rx_clear).pack(fill="x", pady=2)

    def _toggle_auto_mode(self) -> None:
        self.rx_mode_combo.configure(
            state="disabled" if self.auto_mode_var.get() else "readonly")

    # ------------------------------------------------------------------
    # scrollable sidebar
    # ------------------------------------------------------------------
    def _make_scrollable(self, parent):
        """Wrap a column of controls so it can always be reached.

        The controls are laid out top to bottom and are taller than a short
        screen can show.  Without this, anything below the window edge -- which
        is exactly where the "Save image" and "Clear" buttons sit -- becomes
        permanently unreachable, and the window offers no way to scroll to it.
        Returns the frame the caller should put its controls into.
        """
        tk, ttk = self.tk, self.ttk
        canvas = tk.Canvas(parent, background=_PREVIEW_BG, highlightthickness=0, bd=0,
                           width=10)
        bar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        inner = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_inner(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas(event):
            # Make the inner frame exactly as wide as the canvas so the controls
            # stay full width instead of collapsing to their natural size.
            canvas.itemconfigure(window, width=event.width)

        inner.bind("<Configure>", on_inner)
        canvas.bind("<Configure>", on_canvas)

        def wheel(event):
            if canvas.yview() == (0.0, 1.0):
                return                      # nothing to scroll
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        # Bind to the canvas and its children rather than globally, so the wheel
        # still works normally everywhere else in the window.
        for widget in (canvas, inner):
            widget.bind("<MouseWheel>", wheel)

        def bind_wheel(widget):
            widget.bind("<MouseWheel>", wheel, add="+")
            for child in widget.winfo_children():
                bind_wheel(child)

        bind_wheel(inner)
        inner.bind("<Configure>", lambda e: bind_wheel(inner), add="+")
        self._sidebar_canvas = canvas
        return inner

    # ------------------------------------------------------------------
    # waterfall
    # ------------------------------------------------------------------
    def _waterfall_seconds(self) -> float:
        """History length currently selected, in seconds."""
        text = self.wf_seconds_var.get() if hasattr(self, "wf_seconds_var") else WATERFALL_WINDOWS[1]
        try:
            return float(text.split()[0])
        except (ValueError, IndexError):
            return 6.0

    def _waterfall_span(self) -> tuple[float, float]:
        text = self.wf_span_var.get() if hasattr(self, "wf_span_var") else WATERFALL_SPANS[0]
        try:
            low, high = text.replace("Hz", "").split("-")
            return float(low.strip()), float(high.strip())
        except ValueError:
            return waterfall_module.DEFAULT_LOW_HZ, waterfall_module.DEFAULT_HIGH_HZ

    def _apply_waterfall_settings(self) -> None:
        low, high = self._waterfall_span()
        self.waterfall.configure(seconds=self._waterfall_seconds(), low_hz=low, high_hz=high)
        self._mark_waterfall_dirty()

    def _suggest_waterfall_window(self, silent: bool = False) -> None:
        """Pick a history length that suits the mode currently being received.

        Only touches the display when the suggestion is actually different.
        Re-applying the same window would rebuild the row buffer and wipe the
        history, so a decode finishing would otherwise blank the very waterfall
        the operator was watching.
        """
        name = self.rx_mode_var.get() if not self.auto_mode_var.get() else None
        if name is None and self.rx_result is not None and self.rx_result.mode is not None:
            name = self.rx_result.mode.name
        if name is None:
            name = self.mode_var.get()
        suggestion = WATERFALL_WINDOW_BY_MODE.get(name, "6 seconds")
        if self.wf_seconds_var.get() == suggestion:
            return
        self.wf_seconds_var.set(suggestion)
        self._apply_waterfall_settings()
        if not silent:
            self._set_status(f"Waterfall history set to {suggestion} for {name}")

    def _clear_waterfall(self) -> None:
        self.waterfall.clear()
        try:
            self._waterfall_cursor = self.recorder.captured_bytes()
        except Exception:
            self._waterfall_cursor = 0
        self._mark_waterfall_dirty()

    def _mark_waterfall_dirty(self) -> None:
        self._waterfall_dirty = True

    def _on_waterfall_resize(self, event=None) -> None:
        """Keep the waterfall canvas filling its pane, at any window size.

        The canvas is given a nominal height and allowed to expand, so the pane
        decides how much room it gets; the height is then topped up to that
        allocation.  The guard matters: setting the height makes the canvas
        re-issue a Configure event, and feeding each one straight back in would
        loop.  Recomputing from the current height and only applying a real change
        settles after one step.
        """
        canvas = self.rx_waterfall
        water = canvas.master
        available = water.winfo_height()
        if available <= 1:
            return
        # Everything in the pane except the canvas: the heading and the peak row.
        chrome = 0
        for child in water.winfo_children():
            if child is not canvas:
                chrome += child.winfo_height()
        desired = max(60, available - chrome - 10)
        if abs(desired - canvas.winfo_height()) > 6:
            canvas.configure(height=desired)
        self._mark_waterfall_dirty()

    def _start_waterfall(self) -> None:
        """Begin live updates (called when recording starts)."""
        self.waterfall.clear()
        try:
            self._waterfall_cursor = self.recorder.captured_bytes()
        except Exception:
            self._waterfall_cursor = 0
        self._waterfall_dirty = True
        self._start_live_decode()
        if self._waterfall_job is None:
            self._waterfall_job = self._schedule(WATERFALL_INTERVAL_MS, self._pump_waterfall)

    # ------------------------------------------------------------------
    # progressive decoding
    # ------------------------------------------------------------------
    def _start_live_decode(self) -> None:
        """Begin decoding the incoming transmission, line by line."""
        mode = None if self.auto_mode_var.get() else modes_module.get_mode(self.rx_mode_var.get())
        self._live_notes = []
        self._live_finished = False
        self.rx_image = None
        self.rx_result = None
        self._show_image(self.rx_preview, None)
        self.rx_info.configure(text="Listening...")
        self.save_rx_btn.configure(state="disabled")
        try:
            self._live_decoder = streaming_module.StreamingDecoder(
                self.sample_rate, mode,
                progress=self._on_live_progress,
                on_image=self._on_live_image,
                # Every fourth line is enough for the display to look continuous
                # without rebuilding a bitmap dozens of times a second.
                image_every=4)
        except Exception as exc:
            self._live_decoder = None
            self._set_status(f"Live decoding unavailable: {exc}", _WARN)

    def _on_live_progress(self, progress) -> None:
        """Called from the Tk thread for each block of audio."""
        self._live_progress = progress
        if progress.lines_decoded and self.rx_image is None:
            self.rx_image = self._live_decoder.to_image()
        if not progress.mode:
            self.rx_info.configure(
                text=f"Listening... {progress.seconds:.1f} s\n"
                     f"Waiting for the VIS header")
        else:
            lines = [f"Source: {self._live_source()}",
                     (f"Mode: {progress.mode.name} (VIS {progress.vis_code})"
                      if progress.vis_code is not None
                      else f"Mode: {progress.mode.name} (chosen manually)"),
                     f"Receiving: {progress.lines_decoded} / {progress.total_lines} lines"
                     f"  ({progress.coverage * 100:.0f}%)",
                     f"Heard: {progress.seconds:.1f} s"]
            if progress.frequency_offset_hz:
                lines.append(f"Tuning corrected by {progress.frequency_offset_hz:+.0f} Hz")
            lines.append("Decoding as it arrives" if not progress.finished
                         else "Transmission complete")
            self.rx_info.configure(text="\n".join(lines))

    def _on_live_image(self, image) -> None:
        self.rx_image = image
        self._show_image(self.rx_preview, image)
        self.save_rx_btn.configure(state="normal")

    def _finish_live_decode(self) -> None:
        """Called when recording stops: decode the tail and show the outcome."""
        decoder = self._live_decoder
        if decoder is None:
            return
        try:
            progress = decoder.finalize()
        except Exception as exc:
            self._set_status(f"Live decoding failed: {exc}", _WARN)
            self._live_decoder = None
            return
        self._live_decoder = None
        self._live_finished = True
        self._live_notes = decoder.notes

        if progress.mode is None or progress.lines_decoded == 0:
            self.rx_image = None
            self._show_image(self.rx_preview, None)
            self.rx_info.configure(text="No usable SSTV signal was found in that audio.")
            self.save_rx_btn.configure(state="disabled")
            self._set_status("No SSTV signal found", _WARN)
            return

        result = decoder.result()
        self.rx_result = result
        self.rx_image = result.image if result.image.size else None
        self._show_image(self.rx_preview, self.rx_image)
        self.rx_info.configure(text=self._describe(result, self._live_source()))
        self.save_rx_btn.configure(state="normal" if self.rx_image is not None else "disabled")
        if result.complete:
            self._set_status(
                f"Decoded {result.mode.name} from {self._live_source()}", _OK)
        else:
            self._set_status(
                f"Partly decoded {result.mode.name} "
                f"({result.coverage * 100:.0f}% of the lines)", _WARN)

    def _live_source(self) -> str:
        try:
            return "system sound" if self.recorder.loopback else "microphone"
        except Exception:
            return "recording"

    def _stop_waterfall(self) -> None:
        if self._waterfall_job is not None:
            try:
                self.root.after_cancel(self._waterfall_job)
            except Exception:
                pass
            self._waterfall_job = None
        # One last pull so nothing captured is missing from the display, and so
        # the final lines reach the live decoder before it is finalized.
        self._pump_waterfall(chain=False)
        self._render_waterfall()

    def _pump_waterfall(self, chain: bool = True) -> None:
        """Pull whatever audio has arrived and add it to the display."""
        try:
            if self.recorder.is_recording:
                raw, self._waterfall_cursor = self.recorder.bytes_since(self._waterfall_cursor)
                if raw:
                    samples = (np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0)
                    self.waterfall.feed(samples, self.sample_rate)
                    if self._live_decoder is not None:
                        self._live_decoder.feed(samples)
                    self._waterfall_dirty = True
        except Exception as exc:      # never let the display break recording
            self._set_status(f"Waterfall update failed: {exc}", _WARN)
        if self._waterfall_dirty:
            self._render_waterfall()
        if chain and not self._closing:
            self._waterfall_job = self._schedule(WATERFALL_INTERVAL_MS, self._pump_waterfall)

    def _render_waterfall(self) -> None:
        """Redraw the waterfall canvas from the current history."""
        from PIL import ImageTk

        canvas = self.rx_waterfall
        # Size from the container rather than from the canvas itself.  Before the
        # Receive tab has ever been shown its canvas reports the placeholder size
        # of 1x1, and rendering at that size produces a tiny smeared picture that
        # is never corrected, because nothing triggers a repaint afterwards.
        container = canvas.master
        width = max(240, canvas.winfo_width(), container.winfo_width() - 4)
        height = max(60, canvas.winfo_height())
        low, high = self._waterfall_span()
        try:
            image = self.waterfall.to_image(width=width, height=height,
                                            low_hz=low, high_hz=high,
                                            colormap=self._waterfall_colormap)
            if self.wf_scale_var.get():
                image = waterfall_module.add_frequency_scale(
                    image, low, high, strip=20, minor_hz=100.0)
        except Exception as exc:
            self._set_status(f"Waterfall render failed: {exc}", _WARN)
            return

        photo = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor="nw")
        self._waterfall_photo = photo      # Tk drops the image without a reference
        self._waterfall_dirty = False

        strength = self.waterfall.band_level_db()
        if strength <= waterfall_module.FLOOR_DB + 1.0:
            text, colour = "no signal in band", _MUTED
        elif strength < -55.0:
            text, colour = f"band level {strength:.0f} dBFS (weak)", _WARN
        else:
            text, colour = f"band level {strength:.0f} dBFS", _OK
        peak = self.waterfall.peak_hz()
        if peak:
            text = f"peak {peak:.0f} Hz, {text}"
        self.rx_peak_label.configure(text=text, foreground=colour)

    def _build_waterfall_from_audio(self, samples, sample_rate: int) -> None:
        """Fill the display from a whole recording in one pass.

        A file can be transformed in bulk, which is both faster and shows the
        complete transmission at once rather than only the audio that happened to
        arrive while the window was open.
        """
        try:
            data = np.asarray(samples, dtype=np.float64)
            if sample_rate != self.sample_rate:
                from . import dsp as dsp_module

                data = dsp_module.resample(data, sample_rate, self.sample_rate)
            self.waterfall.clear()
            # Feed in blocks so memory stays bounded and the loop below stays
            # responsive on a long recording.
            block = self.sample_rate * 4
            for start in range(0, data.size, block):
                self.waterfall.feed(data[start : start + block], self.sample_rate)
            self._waterfall_dirty = True
            self._render_waterfall()
        except Exception as exc:
            self._set_status(f"Waterfall could not be built: {exc}", _WARN)

    # ------------------------------------------------------------------
    # devices
    # ------------------------------------------------------------------
    def _refresh_devices(self) -> None:
        try:
            outs = audio_module.list_output_devices()
        except Exception as exc:
            self._set_status(f"Audio devices unavailable: {exc}", _ERROR)
            return
        try:
            ins = audio_module.list_input_devices()
        except Exception:
            ins = []
        try:
            loops = audio_module.list_loopback_devices()
        except Exception:
            loops = []

        self._outputs = outs
        self._inputs = ins
        self._loopbacks = loops
        self.tx_device_combo.configure(values=[d.name for d in outs])
        if outs:
            self.tx_device_combo.current(0)

        # Only offer system-sound listening when there is something to listen to,
        # rather than presenting a control that cannot work.
        if loops:
            self.rx_loopback_radio.configure(state="normal")
            self.rx_loopback_hint.configure(text="")
        else:
            self.rx_loopback_radio.configure(state="disabled")
            if self.rx_source_var.get() == "loopback":
                self.rx_source_var.set("microphone")
            self.rx_loopback_hint.configure(
                text="System-sound listening is unavailable: no output device "
                     "could be opened for capture.")
        self._refresh_source_devices()
        self._update_record_button()

    def _refresh_source_devices(self) -> None:
        """Fill the input list for whichever kind of source is selected."""
        source = self.rx_source_var.get() if hasattr(self, "rx_source_var") else "microphone"
        if source == "loopback":
            self._active_inputs = list(self._loopbacks)
        else:
            self._active_inputs = list(self._inputs)
        self.rx_device_combo.configure(values=[d.name for d in self._active_inputs])
        if self._active_inputs:
            self.rx_device_combo.current(0)
        self.rx_device_combo.configure(state="readonly" if self._active_inputs else "disabled")

    def _on_source_changed(self) -> None:
        self._refresh_source_devices()
        self._update_record_button()
        if self.rx_source_var.get() == "loopback":
            self._set_status("Ready to decode whatever this computer is playing")

    def _update_record_button(self) -> None:
        source = self.rx_source_var.get() if hasattr(self, "rx_source_var") else "microphone"
        if source == "loopback":
            label = "\u25cf  Listen to system sound"
        else:
            label = "\u25cf  Record from microphone"
        if self.recorder.is_recording:
            label = "\u25a0  Stop recording"
        self.rx_record_btn.configure(text=label)

    def _selected_output(self) -> int:
        index = self.tx_device_combo.current()
        if 0 <= index < len(self._outputs):
            return self._outputs[index].index
        return -1

    def _selected_input(self) -> int:
        index = self.rx_device_combo.current()
        if 0 <= index < len(self._active_inputs):
            return self._active_inputs[index].index
        return -1

    def _selected_loopback(self):
        """The chosen system-sound device, or None."""
        index = self.rx_device_combo.current()
        if 0 <= index < len(self._loopbacks):
            return self._loopbacks[index]
        return None

    # ------------------------------------------------------------------
    # worker
    # ------------------------------------------------------------------
    def _start_worker(self) -> None:
        thread = threading.Thread(target=self._worker_loop, daemon=True, name="sstv-worker")
        thread.start()

    def _worker_loop(self) -> None:
        while not self._closing:
            try:
                task = self._tasks.get(timeout=0.2)
            except queue.Empty:
                continue
            self._worker_busy = True
            try:
                task.fn()
            except Exception as exc:  # surface, never crash the thread
                detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                self._post("error", f"{task.name or 'Task'} failed: {detail}")
            finally:
                self._worker_busy = False
                self._post("busy", False)

    def _submit(self, fn: Callable[[], None], name: str = "") -> None:
        self._post("busy", True)
        self._tasks.put(_Task(fn, name))

    def _post(self, kind: str, *payload) -> None:
        """Queue one event for the Tk thread.

        A single tuple argument is unwrapped, so ``_post("status", (text, colour))``
        and ``_post("status", text, colour)`` mean the same thing.  Without that,
        the former would arrive as a one-element payload holding the real fields
        and every handler would unpack it wrong -- a mistake that is silent until
        the event fires at run time.
        """
        if len(payload) == 1 and isinstance(payload[0], (tuple, list)):
            payload = tuple(payload[0])
        self._events.put((kind, payload))

    def _pump_events(self) -> None:
        if self._closing:
            return
        try:
            while True:
                kind, payload = self._events.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        self._schedule(80, self._pump_events)

    def _schedule(self, delay_ms: int, callback: Callable[..., None]) -> str | None:
        """Queue a Tk timer unless the window is going away.

        Every repeating timer goes through here.  Rescheduling from a callback
        that runs after ``destroy()`` produces a stream of "invalid command name"
        errors from the Tcl interpreter, which look alarming and are entirely
        avoidable.
        """
        if self._closing:
            return None
        try:
            return self.root.after(delay_ms, callback)
        except Exception:
            return None

    def _handle_event(self, kind: str, payload) -> None:
        """Apply one worker event.

        *payload* is always a tuple whose fields are unpacked positionally, so a
        missing field fails loudly here rather than silently doing nothing.  Every
        ``_post`` call site must supply the same number of fields as its branch
        here reads.
        """
        if kind == "status":
            text, colour = payload
            self._set_status(text, colour)
        elif kind == "progress":
            (value,) = payload
            self.progress["value"] = max(0, min(1000, int(value * 1000)))
        elif kind == "error":
            (message,) = payload
            self._set_status(message, _ERROR)
            self.progress["value"] = 0
        elif kind == "tx_preview":
            (image,) = payload
            self._show_image(self.tx_preview, image)
        elif kind == "tx_info":
            (text,) = payload
            self.tx_info.configure(text=text)
        elif kind == "tx_state":
            (state,) = payload
            self.tx_progress_label.configure(text=state)
            sending = state not in ("Idle", "Ready", "Finished", "Stopped", "Error")
            self.tx_stop_btn.configure(state="normal" if sending else "disabled")
        elif kind == "tx_done":
            message, colour = payload
            self._set_status(message, colour)
            self.progress["value"] = 0
            self.tx_stop_btn.configure(state="disabled")
        elif kind == "busy":
            (busy,) = payload
            self.root.configure(cursor="" if busy else "")
        elif kind == "rx_image":
            (image,) = payload
            self._show_image(self.rx_preview, image)
        elif kind == "rx_info":
            (text,) = payload
            self.rx_info.configure(text=text)
        elif kind == "rx_level":
            fraction, seconds = payload
            self.rx_level["value"] = max(0, min(100, int(fraction * 100)))
            if fraction > 0.97:
                self.rx_level_label.configure(text=f"Level: clipping  ({seconds:.1f} s)",
                                              foreground=_ERROR)
            elif fraction > 0.05:
                self.rx_level_label.configure(text=f"Level: good  ({seconds:.1f} s)",
                                              foreground=_OK)
            else:
                self.rx_level_label.configure(text=f"Level: low  ({seconds:.1f} s)",
                                              foreground=_WARN)
        elif kind == "rx_save_state":
            self.save_rx_btn.configure(state="normal")
        elif kind == "decode_result":
            result, source = payload
            self._apply_decode_result(result, source)
        elif kind == "waterfall_audio":
            samples, rate = payload
            self._build_waterfall_from_audio(samples, rate)
        elif kind == "rx_recording":
            (recording,) = payload
            self._update_record_button()
            if recording:
                self._set_status("Recording...")
            else:
                self._update_record_button()

    def _set_status(self, text: str, colour: str = _MUTED) -> None:
        self.status.configure(text=text, foreground=colour)

    # ------------------------------------------------------------------
    # image display
    # ------------------------------------------------------------------
    def _show_image(self, widget, image) -> None:
        from PIL import Image, ImageTk

        if image is None:
            widget.configure(image="", text="No image")
            widget.image = None
            return
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Fit the picture to its pane, keeping the aspect ratio and using integer
        # magnification so an upscaled picture stays crisp.
        available_w = max(120, widget.winfo_width() - 8)
        available_h = max(90, widget.winfo_height() - 8)
        scale = min(available_w / image.width, available_h / image.height)
        if scale >= 1.0:
            scale = max(1.0, float(int(scale)))
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        if new_size != image.size:
            resample = Image.NEAREST if scale > 1.0 else Image.LANCZOS
            image = image.resize(new_size, resample)
        photo = ImageTk.PhotoImage(image)
        widget.configure(image=photo, text="")
        widget.image = photo  # keep a reference or Tk discards it

    # ------------------------------------------------------------------
    # transmit actions
    # ------------------------------------------------------------------
    def tx_open_image(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Choose an image to transmit",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        try:
            from PIL import Image

            image = Image.open(path)
            image.load()
            self.tx_set_image(image)
            self._set_status(f"Loaded {os.path.basename(path)}")
        except Exception as exc:
            self._set_status(f"Could not open image: {exc}", _ERROR)

    def tx_test_card(self) -> None:
        spec = self._tx_mode()
        self.tx_set_image(encoder_module.make_test_image(*spec.resolution))
        self._set_status("Loaded the built-in test card")

    def tx_from_rx(self) -> None:
        if self.rx_image is None:
            self._set_status("Nothing has been received yet", _WARN)
            return
        from PIL import Image

        self.tx_set_image(Image.fromarray(self.rx_image))
        self._set_status("Copied the received image to the transmitter")

    def tx_set_image(self, image) -> None:
        """Set the image to send and encode it immediately."""
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        if image.mode != "RGB":
            image = image.convert("RGB")
        self.tx_image = image
        self._show_image(self.tx_preview, image)
        mode = self._tx_mode()
        self.tx_info.configure(
            text=f"{image.width} x {image.height} px  \u2192  {mode.name} "
                 f"({mode.image_width} x {mode.image_height})")
        self.tx_audio = None
        self.tx_encode_async()

    def _tx_mode(self):
        return modes_module.get_mode(self.mode_var.get())

    def _update_mode_info(self) -> None:
        mode = self._tx_mode()
        minutes, seconds = divmod(mode.duration_seconds + modes_module.VIS_HEADER_MS / 1000.0, 60)
        self.mode_info.configure(
            text=(f"{mode.image_width} x {mode.image_height} px\n"
                  f"{mode.line_count} lines, {mode.line_milliseconds:.1f} ms each\n"
                  f"transmission time {int(minutes)}m {seconds:04.1f}s\n"
                  f"VIS code {mode.vis_code}")
        )
        if self.tx_image is not None:
            self.tx_set_image(self.tx_image)

    def tx_encode_async(self) -> None:
        image = self.tx_image
        if image is None:
            return
        mode = self._tx_mode()
        self.tx_encoding = True
        self._post("tx_state", "Encoding...")

        def work() -> None:
            started = time.time()
            result = encoder_module.encode(image, mode, self.sample_rate)
            self.tx_audio = result.samples
            self.tx_encoding = False
            self._post("tx_info",
                       f"{image.width} x {image.height} px  \u2192  {mode.name}   "
                       f"{result.duration:.1f} s   encoded in {time.time() - started:.1f} s")
            self._post("tx_state", "Ready")
            self._post("tx_done", f"Ready to transmit {mode.name}", _OK)

        self._submit(work, f"Encoding {mode.name}")

    def tx_transmit(self) -> None:
        if self.player.is_playing:
            self._set_status("Already transmitting", _WARN)
            return
        if self.tx_audio is None:
            self._set_status("Still encoding - please wait a moment", _WARN)
            return
        samples = self.tx_audio
        mode = self._tx_mode()
        device = self._selected_output()

        def work() -> None:
            player = audio_module.AudioPlayer(self.sample_rate, device)
            player.set_callbacks(
                on_progress=lambda f: self._post("progress", f),
                on_finish=lambda state: self._post("tx_done", (
                    "Transmission finished" if state == "idle" else f"Transmission {state}",
                    _OK if state == "idle" else _WARN)),
            )
            self.player = player
            self._post("tx_state", f"Transmitting {mode.name}...")
            try:
                player.play(samples)
            except Exception as exc:
                self._post("tx_done", (f"Could not play: {exc}", _ERROR))
                return
            player.wait()

        self._submit(work, "Transmitting")

    def tx_stop(self) -> None:
        self.player.stop()
        self._post("tx_state", "Stopped")
        self._post("tx_done", ("Transmission stopped", _WARN))

    def tx_save_wav(self) -> None:
        if self.tx_audio is None:
            self._set_status("Encode an image first", _WARN)
            return
        from tkinter import filedialog

        mode = self._tx_mode()
        path = filedialog.asksaveasfilename(
            title="Save the transmission as a WAV file",
            defaultextension=".wav",
            initialfile=f"{mode.name.replace(' ', '')}.wav",
            filetypes=[("WAV audio", "*.wav")],
        )
        if not path:
            return
        try:
            audio_module.write_wav(path, self.tx_audio, self.sample_rate)
            self._set_status(f"Saved {os.path.basename(path)}", _OK)
        except Exception as exc:
            self._set_status(f"Could not save: {exc}", _ERROR)

    # ------------------------------------------------------------------
    # receive actions
    # ------------------------------------------------------------------
    def rx_toggle_record(self) -> None:
        if self.recorder.is_recording:
            self.recorder.stop()
            self._post("rx_recording", False)
            # Let the display take the last of the audio, then close out the
            # progressive decode.  The picture is normally already complete by
            # now, so stopping is instant instead of starting a long decode.
            self._stop_waterfall()
            self._finish_live_decode()
            return
        self.rx_clear(full=False)

        source = self.rx_source_var.get()
        loopback = source == "loopback"
        device = self._selected_loopback() if loopback else None
        if loopback and device is None:
            self._set_status("No system-sound device is available to listen to", _WARN)
            return
        try:
            if loopback:
                self.recorder = audio_module.AudioRecorder(
                    self.sample_rate, loopback=True, device_id=device.device_id)
            else:
                self.recorder = audio_module.AudioRecorder(
                    self.sample_rate, self._selected_input())
            self.recorder.set_level_callback(
                lambda seconds, peak: self._post("rx_level", (peak, seconds)))
            self.recorder.start()
        except Exception as exc:
            what = "system sound" if loopback else "microphone"
            self._set_status(f"Could not open the {what}: {exc}", _ERROR)
            return

        # The device is opened on a worker thread, so it is not running yet.  A
        # capture that is still opening silently misses the beginning of a
        # transmission, and the VIS header -- the part that says which mode is
        # arriving -- lives right there, so a moment's impatience costs the whole
        # picture.  Confirmation is deferred rather than waited for here, to keep
        # the window responsive.
        self.root.after(60, self._check_capture_started)
        self._post("rx_recording", True)
        self._start_waterfall()
        if loopback:
            self._set_status(f"Listening to {device.name}")
        else:
            self._set_status("Opening the microphone...")

    def _check_capture_started(self) -> None:
        """Confirm the capture opened, or report why it did not.

        Scheduled shortly after recording starts.  The wait is bounded and
        generous: opening a device takes tens of milliseconds, and giving up too
        early would abandon a capture that is about to work.
        """
        if self._closing:
            return
        if self.recorder.error:
            self._set_status(self.recorder.error, _ERROR)
            self.recorder.stop()
            self._post("rx_recording", False)
            self._stop_waterfall()
            return
        if self.recorder.wait_until_ready(1.5):
            source = "system sound" if self.recorder.loopback else "the microphone"
            self._set_status(f"Listening to {source}")
            return
        self._set_status("The capture device did not start in time", _WARN)
        self.recorder.stop()
        self._post("rx_recording", False)
        self._stop_waterfall()

    def _decode_recorded(self) -> None:
        """Decode whatever was just captured.

        The samples are read on the Tk thread before handing off, because the
        recorder belongs to the window; the decode itself then runs on the worker
        like any other.
        """
        source = "system sound" if self.recorder.loopback else "microphone"
        samples = self.recorder.samples()
        if not samples:
            self._set_status("Nothing was recorded", _WARN)
            self.rx_info.configure(text="Nothing was recorded.")
            return
        self._decode_samples(samples, source=source)

    def rx_open_file(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Choose a recording or a received audio file",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return
        self._set_status(f"Loading {os.path.basename(path)}...")

        def work() -> None:
            try:
                samples, rate = audio_module.read_wav(path)
            except Exception as exc:
                self._post("status", (f"Could not read the file: {exc}", _ERROR))
                return
            self._post("waterfall_audio", (np.asarray(samples, dtype=np.float64), rate))
            if rate != self.sample_rate:
                from . import dsp

                samples = dsp.resample(np.asarray(samples, dtype=np.float64),
                                       rate, self.sample_rate)
            self._decode_samples(list(samples), source=os.path.basename(path),
                                 samples_array=np.asarray(samples, dtype=np.float64))

        self._submit(work, "Loading audio")

    def _decode_samples(self, samples, source: str = "", samples_array=None) -> None:
        """Kick off a decode.  Must be called from the Tk thread.

        The worker only computes; every assignment to the window's state happens
        back on the Tk thread when the result arrives.  Touching the interface --
        or in fact any Tk object -- from the worker raises "main thread is not in
        main loop", and assigning ``self.rx_image`` there would be a race even if
        it did not.
        """
        array = (np.asarray(samples, dtype=np.float64) if samples_array is None
                 else samples_array)
        if array.size == 0:
            self._post("status", ("Empty recording", _WARN))
            return
        mode = None if self.auto_mode_var.get() else modes_module.get_mode(self.rx_mode_var.get())

        def work() -> None:
            result = decoder_module.decode(
                array, self.sample_rate, mode,
                progress=lambda f: self._post("progress", f),
                on_image=lambda img: self._post("rx_image", (img,)),
            )
            self._post("decode_result", (result, source))

        self._submit(work, "Decoding")

    def _apply_decode_result(self, result, source: str) -> None:
        """Adopt a finished decode, on the Tk thread."""
        self.rx_result = result
        self.rx_image = result.image if result.image.size else None
        self._show_image(self.rx_preview, result.image if result.image.size else None)
        self.rx_info.configure(text=self._describe(result, source))
        self.progress["value"] = 0
        if result.image.size == 0:
            self._set_status("No SSTV signal found", _WARN)
        elif result.complete:
            self._set_status(f"Decoded {result.mode.name} from {source}", _OK)
        else:
            self._set_status(
                f"Partly decoded {result.mode.name} "
                f"({result.coverage * 100:.0f}% of the lines)", _WARN)
        self.save_rx_btn.configure(state="normal")
        if self.rx_image is not None:
            self._suggest_waterfall_window(silent=True)

    @staticmethod
    def _describe(result: decoder_module.DecodeResult, source: str) -> str:
        if not result.image.size:
            return "No usable SSTV signal was found in that audio."
        lines = [f"Source: {source}" if source else ""]
        if result.mode is not None:
            lines.append(f"Mode: {result.mode.name} (VIS {result.vis_code})")
        lines.append(f"Image: {result.image.shape[1]} x {result.image.shape[0]} px")
        lines.append(f"Lines decoded: {result.lines_decoded} / {result.total_lines}"
                     f"  ({result.coverage * 100:.0f}%)")
        if result.vis_confidence:
            lines.append(f"Header confidence: {result.vis_confidence * 100:.0f}%")
        if result.sync_error_hz:
            lines.append(f"Sync accuracy: within {result.sync_error_hz:.0f} Hz")
        if abs(result.frequency_offset_hz) >= 1.0:
            lines.append(f"Tuning offset corrected: {result.frequency_offset_hz:+.0f} Hz")
        if result.notes:
            lines.append("Notes: " + "; ".join(result.notes))
        return "\n".join(line for line in lines if line)

    def rx_save_image(self) -> None:
        if self.rx_image is None:
            return
        from tkinter import filedialog
        from PIL import Image

        path = filedialog.asksaveasfilename(
            title="Save the received image",
            defaultextension=".png",
            initialfile="received.png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")],
        )
        if not path:
            return
        try:
            Image.fromarray(self.rx_image).save(path)
            self._set_status(f"Saved {os.path.basename(path)}", _OK)
        except Exception as exc:
            self._set_status(f"Could not save: {exc}", _ERROR)

    def rx_clear(self, full: bool = True) -> None:
        if full:
            self.rx_image = None
            self.rx_result = None
        self.rx_preview.configure(image="", text="Nothing received yet")
        self.rx_preview.image = None
        if full:
            self.rx_info.configure(text="")
            self.rx_level["value"] = 0
            self.rx_level_label.configure(text="Level", foreground=_MUTED)
            self.save_rx_btn.configure(state="disabled")

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._closing = True
        if self._waterfall_job is not None:
            try:
                self.root.after_cancel(self._waterfall_job)
            except Exception:
                pass
            self._waterfall_job = None
        try:
            self.player.stop()
        except Exception:
            pass
        try:
            self.recorder.stop()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def run(sample_rate: int = audio_module.DEFAULT_SAMPLE_RATE) -> int:
    """Create the window and run the Tk event loop."""
    import tkinter as tk

    root = tk.Tk()
    SstvApp(root, sample_rate)
    root.mainloop()
    return 0


main = run


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
