# ============================================================
# LSTM Text Generation - Shakespeare Dataset
# Author: Harsh
# Description: Train an LSTM model to generate Shakespeare-like text
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import random
import os

# ============================================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================================

print("Loading dataset...")

# Load the Shakespeare text file
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Convert to lowercase and keep only basic characters
text = text.lower()
text = ''.join(c for c in text if c.isalpha() or c in [' ', '\n'])

print(f"Total characters in dataset: {len(text)}")

# Get all unique characters
chars = sorted(set(text))
print(f"Total unique characters: {len(chars)}")

# Create character to index mapping and vice versa
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
idx_to_char = {idx: ch for idx, ch in enumerate(chars)}

# ============================================================
# STEP 2: PREPARE INPUT-OUTPUT SEQUENCES
# ============================================================

print("Preparing sequences...")

SEQ_LENGTH = 50   # Input sequence length
STEP = 3          # Step size for sliding window

input_sequences = []
output_chars = []

# Sliding window to create input-output pairs
for i in range(0, len(text) - SEQ_LENGTH, STEP):
    input_sequences.append(text[i:i + SEQ_LENGTH])
    output_chars.append(text[i + SEQ_LENGTH])

print(f"Total sequences created: {len(input_sequences)}")

# Convert to numpy arrays for model input
print("Vectorizing sequences...")

X = np.zeros((len(input_sequences), SEQ_LENGTH, len(chars)), dtype=np.bool_)
y = np.zeros((len(input_sequences), len(chars)), dtype=np.bool_)

for i, seq in enumerate(input_sequences):
    for t, char in enumerate(seq):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[output_chars[i]]] = 1

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ============================================================
# STEP 3: TRAIN-VALIDATION SPLIT
# ============================================================

split = int(0.9 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# ============================================================
# STEP 4: BUILD LSTM MODEL
# ============================================================

print("Building model...")

model = Sequential([
    # LSTM layer 1
    LSTM(128, input_shape=(SEQ_LENGTH, len(chars)), return_sequences=True),
    
    # LSTM layer 2
    LSTM(128),
    
    # Output layer with softmax for character prediction
    Dense(len(chars), activation='softmax')
])

# Compile with categorical crossentropy loss and Adam optimizer
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# STEP 5: TRAIN THE MODEL
# ============================================================

print("Training model...")

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Save best model checkpoint
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=128,
    epochs=10,
    callbacks=[early_stopping, checkpoint]
)

# ============================================================
# STEP 6: TEXT GENERATION FUNCTION
# ============================================================

def generate_text(seed_text, length=200, temperature=1.0):
    """
    Generate text using trained LSTM model.
    
    Args:
        seed_text: Starting text for generation
        length: Number of characters to generate
        temperature: Controls randomness (lower = more conservative)
    
    Returns:
        Generated text string
    """
    
    # Ensure seed is correct length
    seed_text = seed_text.lower()
    seed_text = ''.join(c for c in seed_text if c.isalpha() or c in [' ', '\n'])
    
    if len(seed_text) < SEQ_LENGTH:
        seed_text = seed_text.ljust(SEQ_LENGTH)
    else:
        seed_text = seed_text[-SEQ_LENGTH:]
    
    generated = seed_text
    current_seq = seed_text
    
    for _ in range(length):
        # Vectorize current sequence
        x_pred = np.zeros((1, SEQ_LENGTH, len(chars)))
        for t, char in enumerate(current_seq):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1
        
        # Predict next character
        preds = model.predict(x_pred, verbose=0)[0]
        
        # Apply temperature sampling
        preds = np.log(preds + 1e-10) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_idx = np.random.choice(len(chars), p=preds)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
        current_seq = current_seq[1:] + next_char
    
    return generated

# ============================================================
# STEP 7: GENERATE SAMPLE OUTPUTS
# ============================================================

print("\n" + "="*60)
print("GENERATING TEXT SAMPLES")
print("="*60)

# Different seed inputs
seeds = [
    "to be or not to be",
    "shall i compare thee",
    "all the world is a stage"
]

# Different temperatures for variety
temperatures = [0.5, 1.0, 1.2]

for seed in seeds:
    for temp in temperatures:
        print(f"\nSeed: '{seed}' | Temperature: {temp}")
        print("-" * 40)
        generated = generate_text(seed, length=200, temperature=temp)
        print(generated)
        print()

print("Done! Model trained and text generated successfully.")