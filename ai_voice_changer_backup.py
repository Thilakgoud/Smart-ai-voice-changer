
"""
AI SMART VOICE CHANGER
Real-time DSP voice transformation with automatic experimental
recording, performance measurements and graph generation.

Install:
pip install numpy scipy sounddevice matplotlib
"""

import time
import csv
import threading
import queue
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 44100
CHANNELS = 1
BLOCK_SIZE = 2048

RESULTS_DIR = Path(__file__).resolve().parent / "Voice_Changer_Experimental_Results"


# ============================================================
# DSP BLOCKS
# ============================================================

class StatefulFilter:

    def __init__(self, order, cutoff, btype):
        self.sos = signal.butter(
            order,
            cutoff,
            btype=btype,
            fs=SAMPLE_RATE,
            output="sos"
        )
        self.zi = np.zeros((self.sos.shape[0], 2))

    def process(self, x):
        y, self.zi = signal.sosfilt(
            self.sos,
            x,
            zi=self.zi
        )
        return y.astype(np.float32)


class PitchShifter:

    def __init__(self, semitones, window=1536):
        self.ratio = 2 ** (semitones / 12)
        self.W = window
        self.H = window + 4
        self.hist = np.zeros(self.H, np.float32)
        self.phase = 0.0

    def process(self, x):
        n = len(x)

        ext = np.concatenate(
            [self.hist, x]
        ).astype(np.float64)

        step = (1.0 - self.ratio) / self.W

        p1 = (
            self.phase + step * np.arange(n)
        ) % 1.0

        self.phase = (
            self.phase + step * n
        ) % 1.0

        base = self.H + np.arange(n)
        out = np.zeros(n)

        for p in (p1, (p1 + 0.5) % 1.0):

            pos = base - (2.0 + self.W * p)

            i0 = np.floor(pos).astype(int)
            frac = pos - i0

            tap = (
                ext[i0] * (1 - frac)
                + ext[i0 + 1] * frac
            )

            out += tap * (
                1.0 - np.abs(2.0 * p - 1.0)
            )

        self.hist = ext[-self.H:].astype(np.float32)

        return out.astype(np.float32)


def compressor(x, threshold=0.35, slope=0.35):

    a = np.abs(x)

    return np.where(
        a > threshold,
        np.sign(x) * (
            threshold + (a - threshold) * slope
        ),
        x
    ).astype(np.float32)


# ============================================================
# VOICE MODES
# ============================================================

VOICES = {

    "Human": dict(hp=80, lp=12000),

    "Girl": dict(hp=100, st=3, lp=13000),

    "Male": dict(hp=60, st=-3, lp=10000),

    "Child": dict(hp=120, st=5, lp=14000),

    "Robot": dict(
        hp=100,
        am=(70, 0.25, 0.75),
        drive=1.4
    ),

    "Alien": dict(
        hp=100,
        am=(5, 0.15, 0.85),
        st=2
    ),

    "Radio": dict(
        bp=(250, 3500),
        drive=1.2
    ),

    "Monster": dict(
        hp=50,
        st=-5,
        drive=1.5
    ),

    "Chipmunk": dict(
        hp=120,
        st=7,
        lp=15000
    )
}


class VoiceEngine:

    def __init__(self):
        self.set_voice("Human")

    def set_voice(self, name):

        c = VOICES[name]

        self.cfg = c

        self.hp = (
            StatefulFilter(2, c["hp"], "highpass")
            if "hp" in c else None
        )

        self.lp = (
            StatefulFilter(2, c["lp"], "lowpass")
            if "lp" in c else None
        )

        self.bp = (
            StatefulFilter(3, list(c["bp"]), "bandpass")
            if "bp" in c else None
        )

        self.ps = (
            PitchShifter(c["st"])
            if "st" in c else None
        )

        self.n0 = 0

    def process(self, x):

        c = self.cfg

        if self.hp:
            x = self.hp.process(x)

        if self.bp:
            x = self.bp.process(x)

        if "am" in c:

            fm, depth, offset = c["am"]

            t = (
                self.n0 + np.arange(len(x))
            ) / SAMPLE_RATE

            x = (
                x * (
                    offset + depth * np.sin(
                        2 * np.pi * fm * t
                    )
                )
            ).astype(np.float32)

        self.n0 += len(x)

        if self.ps:
            x = self.ps.process(x)

        if self.lp:
            x = self.lp.process(x)

        if "drive" in c:
            x = np.tanh(
                x * c["drive"]
            ).astype(np.float32)

        x = compressor(x)

        return np.clip(
            x * 1.5,
            -0.95,
            0.95
        ).astype(np.float32)


# ============================================================
# EXPERIMENTAL RESULTS SYSTEM
# ============================================================

class ResultsRecorder:

    def __init__(self):

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.audio_queue = queue.Queue(maxsize=2000)

        self.active = False
        self.worker = None

        self.session_folder = None

        self.dropped_blocks = 0

    def start(self):

        if self.active:
            return

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.session_folder = (
            RESULTS_DIR / f"Session_{timestamp}"
        )

        self.session_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.dropped_blocks = 0
        self.active = True

        self.worker = threading.Thread(
            target=self._save_worker,
            daemon=True
        )

        self.worker.start()

        print("[RESULTS] Recording started.")

    def add_audio(
        self,
        original,
        processed,
        processing_time,
        voice
    ):

        if not self.active:
            return

        try:

            self.audio_queue.put_nowait(
                (
                    voice,
                    original.copy(),
                    processed.copy(),
                    processing_time
                )
            )

        except queue.Full:

            self.dropped_blocks += 1

    def stop(self):

        if not self.active:
            return

        self.active = False

        self.audio_queue.put(None)

        if self.worker:
            self.worker.join()

        self.worker = None

        print(
            "[RESULTS] Results saved:",
            self.session_folder
        )

    def _save_worker(self):

        recordings = {}

        while True:

            item = self.audio_queue.get()

            if item is None:
                break

            voice, original, processed, elapsed = item

            if voice not in recordings:

                recordings[voice] = {
                    "original": [],
                    "processed": [],
                    "times": []
                }

            recordings[voice]["original"].append(original)
            recordings[voice]["processed"].append(processed)
            recordings[voice]["times"].append(elapsed)

        summary = []

        for voice, data in recordings.items():

            original = np.concatenate(
                data["original"]
            )

            processed = np.concatenate(
                data["processed"]
            )

            times = np.asarray(
                data["times"],
                dtype=np.float64
            )

            folder = self.session_folder / voice

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

            self._save_wav(
                folder / "original.wav",
                original
            )

            self._save_wav(
                folder / "processed.wav",
                processed
            )

            duration = len(original) / SAMPLE_RATE

            input_rms = self._rms(original)
            output_rms = self._rms(processed)

            input_peak = float(
                np.max(np.abs(original))
            )

            output_peak = float(
                np.max(np.abs(processed))
            )

            input_pitch = self._pitch(original)
            output_pitch = self._pitch(processed)

            avg_time = float(
                np.mean(times) * 1000
            )

            max_time = float(
                np.max(times) * 1000
            )

            row = {

                "Voice_Mode": voice,

                "Duration_Seconds": duration,

                "Input_RMS": input_rms,

                "Output_RMS": output_rms,

                "Input_Peak": input_peak,

                "Output_Peak": output_peak,

                "Input_Pitch_Hz_Approx": input_pitch,

                "Output_Pitch_Hz_Approx": output_pitch,

                "Average_Processing_ms": avg_time,

                "Maximum_Processing_ms": max_time,

                "Block_Budget_ms": (
                    BLOCK_SIZE / SAMPLE_RATE
                ) * 1000,

                "Number_Of_Blocks": len(times)
            }

            summary.append(row)

            self._plots(
                original,
                processed,
                voice,
                folder,
                input_rms,
                output_rms
            )

            with open(
                folder / "measurements.txt",
                "w",
                encoding="utf-8"
            ) as f:

                for key, value in row.items():
                    f.write(f"{key}: {value}\n")

        if summary:

            csv_path = (
                self.session_folder /
                "experiment_summary.csv"
            )

            with open(
                csv_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=list(summary[0].keys())
                )

                writer.writeheader()
                writer.writerows(summary)

        with open(
            self.session_folder / "session_notes.txt",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "Results generated from actual microphone recordings.\n"
            )

            f.write(
                "Use the same sentence and recording conditions for each mode.\n"
            )

            f.write(
                "Pitch estimates are approximate and may be unreliable "
                "for Robot and Radio effects.\n"
            )

            f.write(
                "Processing time is DSP time per block, not "
                "end-to-end microphone-to-speaker latency.\n"
            )

            f.write(
                f"Dropped audio blocks: {self.dropped_blocks}\n"
            )

    @staticmethod
    def _save_wav(path, audio):

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        pcm = np.int16(
            np.clip(audio, -1, 1) * 32767
        )

        wavfile.write(
            str(path),
            SAMPLE_RATE,
            pcm
        )

    @staticmethod
    def _rms(audio):

        return float(
            np.sqrt(
                np.mean(
                    np.square(
                        audio.astype(np.float64)
                    )
                )
            )
        )

    @staticmethod
    def _pitch(audio):

        audio = np.asarray(
            audio,
            dtype=np.float64
        )

        if len(audio) < SAMPLE_RATE // 5:
            return ""

        width = min(
            len(audio),
            SAMPLE_RATE
        )

        start = max(
            0,
            (len(audio) - width) // 2
        )

        segment = audio[
            start:start + width
        ]

        segment = segment - np.mean(segment)

        if np.max(np.abs(segment)) < 1e-5:
            return ""

        segment *= np.hanning(len(segment))

        corr = signal.correlate(
            segment,
            segment,
            mode="full",
            method="fft"
        )

        corr = corr[len(corr) // 2:]

        min_lag = int(SAMPLE_RATE / 400)
        max_lag = min(
            int(SAMPLE_RATE / 70),
            len(corr) - 1
        )

        if max_lag <= min_lag:
            return ""

        region = corr[min_lag:max_lag + 1]

        lag = np.argmax(region) + min_lag

        if corr[lag] / (corr[0] + 1e-12) < 0.10:
            return ""

        pitch = SAMPLE_RATE / lag

        if 70 <= pitch <= 400:
            return float(pitch)

        return ""

    def _plots(
        self,
        original,
        processed,
        voice,
        folder,
        input_rms,
        output_rms
    ):

        length = min(
            len(original),
            len(processed),
            SAMPLE_RATE * 10
        )

        if length < 2:
            return

        t = np.arange(length) / SAMPLE_RATE

        # 1. Waveform
        fig, axes = plt.subplots(
            2, 1,
            figsize=(12, 7)
        )

        axes[0].plot(
            t,
            original[:length],
            linewidth=0.6
        )

        axes[0].set_title(
            "Original Audio Waveform"
        )

        axes[1].plot(
            t,
            processed[:length],
            linewidth=0.6
        )

        axes[1].set_title(
            f"{voice} Processed Waveform"
        )

        for ax in axes:
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)

        axes[1].set_xlabel("Time (seconds)")

        fig.tight_layout()

        fig.savefig(
            folder / "waveform_comparison.png",
            dpi=180
        )

        plt.close(fig)

        # 2. FFT
        def fft_data(x):

            x = x[:SAMPLE_RATE * 10]

            window = np.hanning(len(x))

            magnitude = np.abs(
                np.fft.rfft(x * window)
            )

            frequencies = np.fft.rfftfreq(
                len(x),
                1 / SAMPLE_RATE
            )

            db = 20 * np.log10(
                magnitude / (
                    np.max(magnitude) + 1e-12
                ) + 1e-12
            )

            return frequencies, db

        f1, s1 = fft_data(original)
        f2, s2 = fft_data(processed)

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.plot(
            f1, s1,
            label="Original"
        )

        ax.plot(
            f2, s2,
            label="Processed"
        )

        ax.set_xlim(0, 10000)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB, normalized)")

        ax.set_title(
            f"{voice} Frequency Spectrum"
        )

        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            folder / "frequency_spectrum.png",
            dpi=180
        )

        plt.close(fig)

        # 3. Spectrogram
        fig, axes = plt.subplots(
            2, 1,
            figsize=(12, 8)
        )

        axes[0].specgram(
            original[:length],
            NFFT=1024,
            Fs=SAMPLE_RATE,
            noverlap=768
        )

        axes[0].set_title(
            "Original Spectrogram"
        )

        axes[1].specgram(
            processed[:length],
            NFFT=1024,
            Fs=SAMPLE_RATE,
            noverlap=768
        )

        axes[1].set_title(
            f"{voice} Processed Spectrogram"
        )

        for ax in axes:

            ax.set_ylim(0, 8000)
            ax.set_ylabel("Frequency (Hz)")

        axes[1].set_xlabel("Time (seconds)")

        fig.tight_layout()

        fig.savefig(
            folder / "spectrogram_comparison.png",
            dpi=180
        )

        plt.close(fig)

        # 4. RMS
        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.bar(
            ["Original", voice],
            [input_rms, output_rms]
        )

        ax.set_ylabel("RMS Amplitude")

        ax.set_title(
            f"{voice} RMS Comparison"
        )

        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            folder / "rms_comparison.png",
            dpi=180
        )

        plt.close(fig)


# ============================================================
# AUDIO I/O
# ============================================================

engine = VoiceEngine()

pending_voice = None
selected_voice = "Human"

stream = None
running = False

block_times = []

recorder = ResultsRecorder()

active_voice = "Human"


def audio_callback(
    indata,
    outdata,
    frames,
    time_info,
    status
):

    global pending_voice
    global active_voice

    if status:
        print("Audio status:", status)

    try:

        if pending_voice is not None:

            engine.set_voice(pending_voice)

            active_voice = pending_voice

            pending_voice = None

        original = indata[:, 0].copy()

        t0 = time.perf_counter()

        processed = engine.process(original)

        elapsed = time.perf_counter() - t0

        block_times.append(elapsed)

        if len(processed) >= frames:

            output = processed[:frames]

        else:

            output = np.pad(
                processed,
                (0, frames - len(processed))
            )

        outdata[:, 0] = output

        # Store actual microphone input and DSP output.
        recorder.add_audio(
            original,
            output,
            elapsed,
            active_voice
        )

    except Exception as e:

        print("Processing error:", e)

        outdata[:, 0] = indata[:, 0]


def start_voice_changer():

    global running
    global stream

    if running:
        return

    try:

        block_times.clear()

        recorder.start()

        stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype="float32",
            latency="low",
            callback=audio_callback
        )

        stream.start()

        running = True

        status_label.config(
            text="🟢 VOICE CHANGER RUNNING"
        )

        start_button.config(
            state="disabled"
        )

        stop_button.config(
            state="normal"
        )

    except Exception as e:

        recorder.stop()

        messagebox.showerror(
            "Microphone / Speaker Error",
            str(e)
        )


def stop_voice_changer():

    global running
    global stream

    was_running = running
    running = False

    try:

        if stream is not None:

            stream.stop()
            stream.close()

            stream = None

    except Exception as e:

        print(e)

    if was_running:

        recorder.stop()

    if block_times:

        a = np.array(block_times) * 1000

        print(
            f"[{selected_voice}] "
            f"blocks={len(a)} "
            f"mean={a.mean():.3f} ms "
            f"max={a.max():.3f} ms "
            f"block budget={(BLOCK_SIZE / SAMPLE_RATE) * 1000:.2f} ms"
        )

    status_label.config(
        text="🔴 VOICE CHANGER STOPPED"
    )

    start_button.config(
        state="normal"
    )

    stop_button.config(
        state="disabled"
    )


def select_voice(name):

    global pending_voice
    global selected_voice

    selected_voice = name
    pending_voice = name

    block_times.clear()

    voice_label.config(
        text=f"Selected Voice: {name}"
    )


# ============================================================
# GUI - ORIGINAL DESIGN
# ============================================================

root = tk.Tk()

root.title("🎙️ AI Voice Changer")
root.geometry("650x650")
root.resizable(False, False)

tk.Label(
    root,
    text="🎙️ AI VOICE CHANGER",
    font=("Arial", 24, "bold")
).pack(pady=20)

tk.Label(
    root,
    text="Real-Time DSP Voice Effects",
    font=("Arial", 12)
).pack()

voice_label = tk.Label(
    root,
    text="Selected Voice: Human",
    font=("Arial", 16, "bold")
)

voice_label.pack(pady=20)

voice_frame = tk.Frame(root)
voice_frame.pack()

buttons = [
    ("👨 Human", "Human"),
    ("👩 Girl", "Girl"),
    ("👨 Male", "Male"),
    ("👶 Child", "Child"),
    ("🤖 Robot", "Robot"),
    ("👽 Alien", "Alien"),
    ("📻 Radio", "Radio"),
    ("👹 Monster", "Monster"),
    ("🐿️ Chipmunk", "Chipmunk")
]

for i, (label, name) in enumerate(buttons):

    tk.Button(
        voice_frame,
        text=label,
        width=18,
        height=2,
        font=("Arial", 11, "bold"),
        command=lambda v=name: select_voice(v)
    ).grid(
        row=i // 3,
        column=i % 3,
        padx=8,
        pady=8
    )

control = tk.Frame(root)
control.pack(pady=25)

start_button = tk.Button(
    control,
    text="🎤 START",
    width=15,
    height=2,
    font=("Arial", 12, "bold"),
    command=start_voice_changer
)

start_button.grid(
    row=0,
    column=0,
    padx=10
)

stop_button = tk.Button(
    control,
    text="🛑 STOP",
    width=15,
    height=2,
    font=("Arial", 12, "bold"),
    command=stop_voice_changer,
    state="disabled"
)

stop_button.grid(
    row=0,
    column=1,
    padx=10
)

status_label = tk.Label(
    root,
    text="🔴 VOICE CHANGER STOPPED",
    font=("Arial", 13, "bold")
)

status_label.pack(pady=10)

tk.Label(
    root,
    justify="left",
    font=("Arial", 11),
    text=(
        "1. Connect headphones (avoid feedback)\n"
        "2. Select a voice\n"
        "3. Click START\n"
        "4. Speak into the microphone\n"
        "5. Click STOP"
    )
).pack(pady=15)


def on_close():

    stop_voice_changer()

    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    on_close
)


if __name__ == "__main__":

    root.mainloop()
