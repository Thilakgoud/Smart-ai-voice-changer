"""
AI SMART VOICE CHANGER - real-time multi-effect voice transformation using DSP.
Single-file version. No neural networks, no datasets: live microphone input only.

Requires: pip install numpy scipy sounddevice   (Tkinter ships with Python)
"""
import time
import tkinter as tk
from tkinter import messagebox

import numpy as np
import sounddevice as sd
from scipy import signal

# ---------------------------------------------------------------- settings
SAMPLE_RATE = 44100
CHANNELS = 1
BLOCK_SIZE = 2048          # 2048 / 44100 = 46.44 ms per block (theoretical)


# ---------------------------------------------------------------- DSP blocks
class StatefulFilter:
    """Butterworth filter (SOS form); state is carried across blocks."""

    def __init__(self, order, cutoff, btype):
        self.sos = signal.butter(order, cutoff, btype=btype,
                                 fs=SAMPLE_RATE, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2))

    def process(self, x):
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32)


class PitchShifter:
    """Two-tap delay-line pitch shifter with triangular cross-fade.
    Output length equals input length; history is carried across blocks."""

    def __init__(self, semitones, window=1536):
        self.ratio = 2 ** (semitones / 12)
        self.W = window
        self.H = window + 4
        self.hist = np.zeros(self.H, np.float32)
        self.phase = 0.0

    def process(self, x):
        n = len(x)
        ext = np.concatenate([self.hist, x]).astype(np.float64)
        step = (1.0 - self.ratio) / self.W
        p1 = (self.phase + step * np.arange(n)) % 1.0
        self.phase = (self.phase + step * n) % 1.0
        base = self.H + np.arange(n)
        out = np.zeros(n)
        for p in (p1, (p1 + 0.5) % 1.0):
            pos = base - (2.0 + self.W * p)          # fractional delay
            i0 = np.floor(pos).astype(int)
            frac = pos - i0
            tap = ext[i0] * (1 - frac) + ext[i0 + 1] * frac
            out += tap * (1.0 - np.abs(2.0 * p - 1.0))   # gains sum to 1
        self.hist = ext[-self.H:].astype(np.float32)
        return out.astype(np.float32)


def compressor(x, threshold=0.35, slope=0.35):
    """Static hard-knee compressor (about 2.9:1 above threshold)."""
    a = np.abs(x)
    return np.where(a > threshold,
                    np.sign(x) * (threshold + (a - threshold) * slope),
                    x).astype(np.float32)


# ---------------------------------------------------------------- voice table
# hp/lp: cutoff Hz | bp: (low, high) Hz | st: semitones
# am: (mod freq Hz, depth, offset) | drive: tanh gain
VOICES = {
    "Human":    dict(hp=80, lp=12000),
    "Girl":     dict(hp=100, st=3, lp=13000),
    "Male":     dict(hp=60, st=-3, lp=10000),
    "Child":    dict(hp=120, st=5, lp=14000),
    "Robot":    dict(hp=100, am=(70, 0.25, 0.75), drive=1.4),
    "Alien":    dict(hp=100, am=(5, 0.15, 0.85), st=2),
    "Radio":    dict(bp=(250, 3500), drive=1.2),
    "Monster":  dict(hp=50, st=-5, drive=1.5),
    "Chipmunk": dict(hp=120, st=7, lp=15000),
}


class VoiceEngine:
    """Per-voice processing chain:
    HPF -> BPF -> modulation -> pitch shift -> LPF -> tanh -> compressor -> gain/limiter"""

    def __init__(self):
        self.set_voice("Human")

    def set_voice(self, name):
        c = VOICES[name]
        self.cfg = c
        self.hp = StatefulFilter(2, c["hp"], "highpass") if "hp" in c else None
        self.lp = StatefulFilter(2, c["lp"], "lowpass") if "lp" in c else None
        self.bp = StatefulFilter(3, list(c["bp"]), "bandpass") if "bp" in c else None
        self.ps = PitchShifter(c["st"]) if "st" in c else None
        self.n0 = 0                       # running sample index (continuous phase)

    def process(self, x):
        c = self.cfg
        if self.hp: x = self.hp.process(x)
        if self.bp: x = self.bp.process(x)
        if "am" in c:
            fm, depth, offset = c["am"]
            t = (self.n0 + np.arange(len(x))) / SAMPLE_RATE
            x = (x * (offset + depth * np.sin(2 * np.pi * fm * t))).astype(np.float32)
        self.n0 += len(x)
        if self.ps: x = self.ps.process(x)
        if self.lp: x = self.lp.process(x)
        if "drive" in c: x = np.tanh(x * c["drive"]).astype(np.float32)
        x = compressor(x)
        return np.clip(x * 1.5, -0.95, 0.95).astype(np.float32)


# ---------------------------------------------------------------- audio I/O
engine = VoiceEngine()
pending_voice = None          # GUI sets this; callback applies it safely
selected_voice = "Human"
stream = None
running = False
block_times = []              # per-block processing time (s), for experiments


def audio_callback(indata, outdata, frames, time_info, status):
    global pending_voice
    if status:
        print("Audio status:", status)
    try:
        if pending_voice is not None:         # switch voice between blocks
            engine.set_voice(pending_voice)
            pending_voice = None
        t0 = time.perf_counter()
        out = engine.process(indata[:, 0].copy())
        block_times.append(time.perf_counter() - t0)
        outdata[:, 0] = out[:frames] if len(out) >= frames else np.pad(out, (0, frames - len(out)))
    except Exception as e:
        print("Processing error:", e)
        outdata[:, 0] = indata[:, 0]


def start_voice_changer():
    global running, stream
    if running:
        return
    try:
        block_times.clear()
        stream = sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                           channels=CHANNELS, dtype="float32",
                           latency="low", callback=audio_callback)
        stream.start()
        running = True
        status_label.config(text="🟢 VOICE CHANGER RUNNING")
        start_button.config(state="disabled")
        stop_button.config(state="normal")
    except Exception as e:
        messagebox.showerror("Microphone / Speaker Error", str(e))


def stop_voice_changer():
    global running, stream
    running = False
    try:
        if stream is not None:
            stream.stop()
            stream.close()
            stream = None
    except Exception as e:
        print(e)
    if block_times:
        a = np.array(block_times) * 1000
        print(f"[{selected_voice}] blocks={len(a)} mean={a.mean():.3f} ms "
              f"max={a.max():.3f} ms (block budget 46.44 ms)")
    status_label.config(text="🔴 VOICE CHANGER STOPPED")
    start_button.config(state="normal")
    stop_button.config(state="disabled")


def select_voice(name):
    global pending_voice, selected_voice
    selected_voice = name
    pending_voice = name
    block_times.clear()
    voice_label.config(text=f"Selected Voice: {name}")


# ---------------------------------------------------------------- GUI
root = tk.Tk()
root.title("🎙️ AI Voice Changer")
root.geometry("650x650")
root.resizable(False, False)

tk.Label(root, text="🎙️ AI VOICE CHANGER", font=("Arial", 24, "bold")).pack(pady=20)
tk.Label(root, text="Real-Time DSP Voice Effects", font=("Arial", 12)).pack()
voice_label = tk.Label(root, text="Selected Voice: Human", font=("Arial", 16, "bold"))
voice_label.pack(pady=20)

voice_frame = tk.Frame(root)
voice_frame.pack()
buttons = [("👨 Human", "Human"), ("👩 Girl", "Girl"), ("👨 Male", "Male"),
           ("👶 Child", "Child"), ("🤖 Robot", "Robot"), ("👽 Alien", "Alien"),
           ("📻 Radio", "Radio"), ("👹 Monster", "Monster"), ("🐿️ Chipmunk", "Chipmunk")]
for i, (label, name) in enumerate(buttons):
    tk.Button(voice_frame, text=label, width=18, height=2, font=("Arial", 11, "bold"),
              command=lambda v=name: select_voice(v)).grid(row=i // 3, column=i % 3, padx=8, pady=8)

control = tk.Frame(root)
control.pack(pady=25)
start_button = tk.Button(control, text="🎤 START", width=15, height=2,
                         font=("Arial", 12, "bold"), command=start_voice_changer)
start_button.grid(row=0, column=0, padx=10)
stop_button = tk.Button(control, text="🛑 STOP", width=15, height=2,
                        font=("Arial", 12, "bold"), command=stop_voice_changer, state="disabled")
stop_button.grid(row=0, column=1, padx=10)

status_label = tk.Label(root, text="🔴 VOICE CHANGER STOPPED", font=("Arial", 13, "bold"))
status_label.pack(pady=10)
tk.Label(root, justify="left", font=("Arial", 11),
         text="1. Connect headphones (avoid feedback)\n2. Select a voice\n3. Click START\n"
              "4. Speak into the microphone\n5. Click STOP").pack(pady=15)


def on_close():
    stop_voice_changer()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)

if __name__ == "__main__":
    root.mainloop()