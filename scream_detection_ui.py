import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import customtkinter as ctk
import time

# ================= CONFIG =================
MODEL_PATH = "scream_cnn_lstm_model.h5"
SAMPLE_RATE = 16000
DURATION = 1
THRESHOLD = 0.7   # 🔥 UPDATED

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)

# ================= UI THEME =================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ================= FEATURE EXTRACTION =================
def extract_features(audio):
    if len(audio) > SAMPLE_RATE:
        audio = audio[:SAMPLE_RATE]
    else:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))

    audio = librosa.util.normalize(audio)

    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128)
    mel = librosa.power_to_db(mel)

    if mel.shape[1] < 128:
        mel = np.pad(mel, ((0,0),(0,128-mel.shape[1])))
    else:
        mel = mel[:, :128]

    mel = mel[:128, :128]

    mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)

    mel = np.expand_dims(mel, axis=-1)
    mel = np.expand_dims(mel, axis=0)

    return mel.astype("float32")

# ================= DETECTION FUNCTION =================
def detect_scream():
    detect_btn.configure(state="disabled")

    status_label.configure(text="🎤 Listening...", text_color="yellow")
    app.update()

    preds = []
    max_amp = 0

    # 🔥 Take 3 samples for stability
    for _ in range(3):
        audio = sd.rec(int(SAMPLE_RATE * DURATION),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype='float32')
        sd.wait()

        audio = audio.flatten()

        # Track max amplitude
        max_amp = max(max_amp, np.max(np.abs(audio)))

        features = extract_features(audio)
        pred = model.predict(features, verbose=0)[0][0]

        print("Prediction sample:", pred)  # DEBUG
        preds.append(pred)

    prediction = np.mean(preds)

    print("Final Prediction:", prediction)
    print("Max amplitude:", max_amp)

    # 🔇 Silence filter
    if max_amp < 0.01:
        status_label.configure(text="🔇 No Sound", text_color="gray")
        confidence_label.configure(text="Confidence: 0.00")
        detect_btn.configure(state="normal")
        return

    status_label.configure(text="🧠 Processing...", text_color="orange")
    app.update()
    time.sleep(0.3)

    # 🎯 Final decision
    if prediction > THRESHOLD:
        status_label.configure(
            text="🚨 SCREAM DETECTED",
            text_color="#ff4d4d"
        )
    else:
        status_label.configure(
            text="✅ No Scream Detected",
            text_color="#4CAF50"
        )

    confidence_label.configure(text=f"Confidence: {prediction:.2f}")

    detect_btn.configure(state="normal")

# ================= UI =================
app = ctk.CTk()
app.geometry("500x350")
app.title("AI Scream Detector")

# Title
title = ctk.CTkLabel(
    app,
    text="🎤 Scream Detector",
    font=("Segoe UI", 26, "bold")
)
title.pack(pady=20)

# Frame
frame = ctk.CTkFrame(app, corner_radius=20)
frame.pack(pady=10, padx=20, fill="both", expand=True)

# Status
status_label = ctk.CTkLabel(
    frame,
    text="Press Detect",
    font=("Segoe UI", 20, "bold")
)
status_label.pack(pady=20)

# Confidence
confidence_label = ctk.CTkLabel(
    frame,
    text="Confidence: 0.00",
    font=("Segoe UI", 16)
)
confidence_label.pack()

# Detect Button
detect_btn = ctk.CTkButton(
    app,
    text="🎯 Detect",
    font=("Segoe UI", 18, "bold"),
    height=50,
    corner_radius=15,
    command=detect_scream
)
detect_btn.pack(pady=20)

# Footer
footer = ctk.CTkLabel(
    app,
    text="AI Real-Time Detection System",
    font=("Segoe UI", 12),
    text_color="gray"
)
footer.pack()

# Run
app.mainloop()