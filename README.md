*AI Scream Detection System*

 Project Overview
 
This project is a real-time scream detection system that identifies human screams or distress sounds using machine learning and audio signal processing.
It captures audio through a microphone, extracts key features, and classifies whether the sound is a Scream or Non-Scream.


Features:

Real-time audio detection using microphone
Machine Learning-based classification
Feature extraction using:
MFCC (Mel-Frequency Cepstral Coefficients)
Chroma Features
Mel Spectrogram
Fast and efficient predictions
GUI-based interface (Tkinter)
Detects high-intensity sounds like screams/shouts


Tech Stack:

Python
TensorFlow / Keras
Librosa
NumPy
Scikit-learn
Tkinter



Project Structure:

scream-detection/
│
├── dataset/              # Scream & Non-scream audio files
├── models/               # Trained model files
├── src/
│   ├── train.py          # Model training script
│   ├── predict.py        # Prediction logic
│   └── features.py       # Feature extraction
│
├── ui/
│   └── app.py            # GUI application
│
├── requirements.txt
├── README.md
└── .gitignore

Installation:

Clone the repository:
git clone https://github.com/your-username/scream-detection.git
cd scream-detection
Install dependencies:
pip install -r requirements.txt

Usage:

Run the GUI:
python ui/app.py
Train the Model:
python src/train.py

Model Details:

Input: Audio signal
Features Used: MFCC, Chroma, Mel Spectrogram
Model: Dense Neural Network (TensorFlow)
Output: Scream / Non-Scream classification

Dataset:

The dataset contains:
Human scream audio samples
Non-scream environmental sounds
(Kaggle)


Future Improvements:

Improve model accuracy with larger dataset
Deploy as a mobile application
Integrate with CCTV/security systems
Add real-time alert/notification system


License:

This project is intended for educational and research purposes.


⭐ Support:
If you found this project useful, consider giving it a ⭐ on GitHub.

