import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import threading
from datetime import datetime
import librosa
import random  # For generating random values

class AudioRecorderGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Recorder with Augmentation")
        self.master.geometry("600x500")
        self.master.minsize(600, 500)
        self.master.columnconfigure(0, weight=1)

        self.output_folder = ""
        self.duration = tk.StringVar(value="3")
        self.is_recording = False
        self.pitch_shift = tk.DoubleVar(value=0)
        self.noise_level = tk.DoubleVar(value=0)
        self.volume = tk.DoubleVar(value=1)

        self.style = ttk.Style()
        self.style.configure("TScale", sliderlength=30)

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.master)
        main_frame.grid(sticky="nsew", padx=20, pady=20)
        main_frame.columnconfigure(0, weight=1)

        # Center-aligned top elements
        ttk.Label(main_frame, text="Recording Duration (seconds):").grid(row=0, column=0, pady=5)
        ttk.Entry(main_frame, textvariable=self.duration, width=10).grid(row=1, column=0, pady=5)

        ttk.Button(main_frame, text="Select Output Folder", command=self.select_folder).grid(row=2, column=0, pady=10)

        self.folder_label = ttk.Label(main_frame, text="No folder selected")
        self.folder_label.grid(row=3, column=0, pady=5)

        # Slider frame for better alignment
        slider_frame = ttk.Frame(main_frame)
        slider_frame.grid(row=4, column=0, sticky="ew")
        slider_frame.columnconfigure(1, weight=1)

        # Left-aligned labels, center-aligned sliders and controls
        self.create_slider(slider_frame, "Pitch Shift", self.pitch_shift, -1.0, 1.0, 0, row=0)
        self.create_slider(slider_frame, "Noise Level", self.noise_level, 0, 0.05, 0, row=1)
        self.create_slider(slider_frame, "Volume", self.volume, 0.5, 2, 1, row=2)

        self.record_button = ttk.Button(main_frame, text="Record", command=self.toggle_recording)
        self.record_button.grid(row=5, column=0, pady=20)

    def create_slider(self, parent, label, variable, from_, to, default, row):
        # Left-aligned label
        ttk.Label(parent, text=f"{label}:", anchor="w").grid(row=row, column=0, sticky="w", padx=(0, 10))
        
        # Center-aligned slider
        slider = ttk.Scale(parent, from_=from_, to=to, variable=variable, 
                           orient="horizontal", length=300, style="TScale")
        slider.grid(row=row, column=1, sticky="ew")
        
        # Center-aligned value label
        value_label = ttk.Label(parent, text=f"{variable.get():.2f}", width=10, anchor="center")
        value_label.grid(row=row, column=2, padx=(10, 0))

        # Random button to generate a random value within the limits
        random_button = ttk.Button(parent, text="Random", 
                                   command=lambda: self.set_random_value(variable, from_, to, value_label))
        random_button.grid(row=row, column=3, padx=(10, 0))

        # Center-aligned reset button
        reset_button = ttk.Button(parent, text="Reset", 
                                  command=lambda: self.reset_slider(variable, default, value_label))
        reset_button.grid(row=row, column=4, padx=(10, 0))

        def update_label(event):
            value_label.config(text=f"{variable.get():.2f}")

        slider.bind("<Motion>", update_label)

    def set_random_value(self, variable, from_, to, label):
        random_value = random.uniform(from_, to)
        variable.set(random_value)
        label.config(text=f"{random_value:.2f}")

    def reset_slider(self, variable, default, label):
        variable.set(default)
        label.config(text=f"{default:.2f}")

    def select_folder(self):
        self.output_folder = filedialog.askdirectory()
        self.folder_label.config(text=f"Folder: {self.output_folder}")

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.output_folder:
            messagebox.showerror("Error", "Please select an output folder first.")
            return

        self.is_recording = True
        self.record_button.config(text="Recording...", state="disabled")
        threading.Thread(target=self.record_audio, daemon=True).start()

    def stop_recording(self):
        self.is_recording = False
        self.record_button.config(text="Record", state="normal")

    def record_audio(self):
        try:
            duration = int(self.duration.get())
            fs = 44100  # Sample rate
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()  # Wait until recording is finished

            # Apply augmentations
            augmented_audio = self.apply_augmentations(recording.flatten(), fs)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            file_path = os.path.join(self.output_folder, filename)
            sf.write(file_path, augmented_audio, fs)

            self.master.after(0, self.stop_recording)
            messagebox.showinfo("Success", f"Recording saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.master.after(0, self.stop_recording)

    def apply_augmentations(self, audio, fs):
        # Pitch shifting
        pitch_shift = self.pitch_shift.get()
        if pitch_shift != 0:
            audio = librosa.effects.pitch_shift(audio, sr=fs, n_steps=pitch_shift)

        # Noise addition
        noise_level = self.noise_level.get()
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, len(audio))
            audio = audio + noise

        # Volume control
        volume = self.volume.get()
        audio = audio * volume

        return audio

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorderGUI(root)
    root.mainloop()
