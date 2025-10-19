import random
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# --- Try importing striprtf safely ---
try:
    from striprtf.striprtf import rtf_to_text
    has_striprtf = True
except ImportError:
    has_striprtf = False
    print("⚠️ striprtf not found. If using .rtf file, run: pip install striprtf")

# --- Add Arabic reshaper & bidi for correct display ---
import arabic_reshaper
from bidi.algorithm import get_display

def display_arabic(text):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

# --- Function to normalize Arabic text ---
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("[ًٌٍَُِّْـ]", "", text)  # remove diacritics
    return text

# --- Load and clean text ---
filepath = r"D:\Amgoda\AI\file.rtf"  # ✅ Your actual file path

if filepath.endswith(".rtf") and has_striprtf:
    with open(filepath, 'r', encoding='utf-8') as f:
        rtf_content = f.read()
    text = rtf_to_text(rtf_content)
else:
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

# --- Normalize Arabic ---
text = normalize_arabic(text)

# --- Prepare characters and mappings ---
characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

SEQ_LENGTH = 40
STEP_SIZE = 5  # fewer samples → faster training

# --- Dataset ---
sentences = []
next_characters = []
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# --- Model (small & fast) ---
model = Sequential([
    LSTM(64, input_shape=(SEQ_LENGTH, len(characters))),
    Dense(len(characters)),
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# --- Train quickly ---
print("\n🚀 Training quickly (5 epochs for test)...")
model.fit(x, y, batch_size=128, epochs=250, verbose=1)

# --- Save model ---
model.save('textgenerator.keras')
print("\n✅ Model saved as textgenerator.keras")

# --- Sampling ---
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# --- Generate text ---
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence

    for _ in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            if character in char_to_index:
                x_pred[0, t, char_to_index[character]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

# --- Generate sample output ---
for temp in [0.2, 0.6, 1.0]:
    print(f"\n---------- Temperature: {temp} ----------\n")
    output = generate_text(300, temp)
    print(display_arabic(output))  # ✅ properly displayed Arabic
