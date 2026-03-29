import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from glob import glob
from sklearn.utils import class_weight

# ================= CONFIG =================
DATA_DIR = "D:/scream detection/dataset"
EPOCHS = 50
BATCH_SIZE = 32

# ================= GPU CHECK =================
print("GPU:", tf.config.list_physical_devices('GPU'))

# ================= LOAD FILES =================
def load_files(base):
    files, labels = [], []
    for label, sub in enumerate(['NotScreaming', 'Screaming']):
        subdir = os.path.join(base, sub)
        found = glob(os.path.join(subdir, '*.wav')) + glob(os.path.join(subdir, '*.mp3'))
        print(f"[INFO] {sub}: {len(found)} files")
        files.extend(found)
        labels.extend([label] * len(found))
    return files, labels

# ================= AUGMENTATION =================
def augment_audio(y, sr):
    if np.random.rand() < 0.5:
        y = librosa.effects.time_stretch(y, rate=0.8 + np.random.rand() * 0.4)

    if np.random.rand() < 0.5:
        noise = 0.01 * np.random.randn(len(y))
        y = y + noise

    if np.random.rand() < 0.3:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.randint(-2, 3))

    return y

# ================= SPEC AUGMENT =================
def spec_augment(spec):
    spec = spec.copy()

    for _ in range(2):
        t = np.random.randint(5, 20)
        t0 = np.random.randint(0, spec.shape[1] - t)
        spec[:, t0:t0+t] = 0

    for _ in range(2):
        f = np.random.randint(5, 20)
        f0 = np.random.randint(0, spec.shape[0] - f)
        spec[f0:f0+f, :] = 0

    return spec

# ================= FEATURE EXTRACTION =================
def wav_to_features(path, training=True):
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)

        # Fix length (1 sec)
        if len(y) > 16000:
            y = y[:16000]
        else:
            y = np.pad(y, (0, 16000 - len(y)))

        y = librosa.util.normalize(y)

        if training:
            y = augment_audio(y, sr)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel = librosa.power_to_db(mel)

        if training:
            mel = spec_augment(mel)

        # Resize to 128x128
        if mel.shape[1] < 128:
            mel = np.pad(mel, ((0,0),(0,128-mel.shape[1])))
        else:
            mel = mel[:, :128]

        mel = mel[:128, :128]

        # Normalize
        mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)

        mel = np.expand_dims(mel, axis=-1)

        return mel.astype("float32")

    except Exception as e:
        print(f"[ERROR] {path}: {e}")
        return None

# ================= DATASET =================
def make_dataset(base, training=True):
    files, labels = load_files(base)

    X, Y = [], []
    for f, l in zip(files, labels):
        feat = wav_to_features(f, training)
        if feat is not None:
            X.append(feat)
            Y.append(l)

    print(f"[INFO] Loaded {len(X)} samples")
    return np.array(X), np.array(Y)

# ================= LOAD DATA =================
print("=== TRAIN DATA ===")
X_train, y_train = make_dataset(os.path.join(DATA_DIR, "train"), training=True)

print("=== VAL DATA ===")
X_val, y_val = make_dataset(os.path.join(DATA_DIR, "val"), training=False)

# ================= CLASS WEIGHTS =================
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ================= MODEL =================
def build_model():
    inputs = keras.Input(shape=(128,128,1))

    # CNN
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)

    # 🔥 FIXED RESHAPE (IMPORTANT)
    x = keras.layers.Reshape((256, 128))(x)

    # LSTM
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True)
    )(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(32)
    )(x)

    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)

    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    return model

model = build_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name="auc")]
)

model.summary()

# ================= CALLBACKS =================
callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)
]

# ================= TRAIN =================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks
)

# ================= EVALUATE =================
loss, acc, auc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {acc:.3f}, AUC: {auc:.3f}")

# ================= SAVE =================
model.save("scream_cnn_lstm_model.h5")
print("✅ Model saved successfully!")