🚁 ResQ-Tarot Drone – AI Scream Detection Module
🚀 Project Overview

This project is a critical module of the ResQ-Tarot Disaster Management Drone, designed to assist in search and rescue operations.

The system uses a dedicated microphone and AI-based audio classification to detect human screams or distress signals. Upon detection, the drone dynamically updates its navigation path toward the sound source, enabling faster victim localization in disaster scenarios.

🎯 Objective

To develop an intelligent audio-based detection system that:

Identifies human screams in real time
Assists autonomous drones in locating victims
Enhances rescue efficiency in disaster environments
⚙️ How It Works
The drone captures environmental audio using a microphone
Audio signals are processed and features are extracted
The trained model classifies:
Scream
Non-Scream
If a scream is detected:
Signal is sent to the drone navigation system
Drone adjusts its path toward the sound source
🎯 Features
🎙️ Real-time audio monitoring
🧠 AI-based scream classification
🔍 Feature extraction:
MFCC
Chroma
Mel Spectrogram
🚁 Integration with autonomous drone navigation
⚡ Low-latency detection for real-time response
🚨 Supports emergency rescue operations
🛠️ Tech Stack
Python
TensorFlow / Keras
Librosa
NumPy
Scikit-learn
Tkinter (for testing UI)
📂 Project Structure
scream-detection/
│
├── dataset/              # Scream & Non-scream audio data
├── models/               # Trained ML models
├── src/
│   ├── train.py          # Training script
│   ├── predict.py        # Prediction logic
│   └── features.py       # Audio feature extraction
│
├── ui/
│   └── app.py            # Testing GUI
│
├── integration/          # Drone integration logic (future scope)
├── requirements.txt
├── README.md
└── .gitignore
▶️ Usage
Run Detection System:
python ui/app.py
Train Model:
python src/train.py
🧪 Model Details
Input: Audio signal from microphone
Features Used:
MFCC
Chroma Features
Mel Spectrogram
Model: Dense Neural Network (TensorFlow)
Output: Scream / Non-Scream
🌍 Application
Disaster management (earthquakes, floods, collapsed buildings)
Search and rescue missions
Smart surveillance systems
Emergency response automation
🔐 Future Enhancements
Directional sound localization (multi-mic array)
Integration with GPS & drone autopilot
Real-time alert system to rescue teams
Edge deployment on embedded systems (Jetson / Raspberry Pi)
👥 Contributors
Your Name
Team Member 1
Team Member 2
📜 License

For educational and research purposes.

⭐ Project Significance

This project contributes to saving lives by enabling faster detection of victims in disaster-hit areas using AI-powered autonomous drones.
