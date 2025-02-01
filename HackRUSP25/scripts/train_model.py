import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def load_features(feature_folder):
    """Load all .npy files from a feature folder."""
    files = [f for f in os.listdir(feature_folder) if f.endswith('.npy')]
    if not files:
        print(f"⚠️ Warning: No feature files found in {feature_folder}")
        return []
    return [np.load(os.path.join(feature_folder, f)) for f in files]

# Load features for both categories
X_stuttering_motion = load_features("data/features/mouth_motion/stuttering")
X_stuttering_mfcc = load_features("data/features/mfcc/stuttering")
X_lisp_motion = load_features("data/features/mouth_motion/lisp")
X_lisp_mfcc = load_features("data/features/mfcc/lisp")

# Ensure all feature sets are aligned in time steps
X_stuttering = []
X_lisp = []
y_stuttering = []
y_lisp = []

for motion, mfcc in zip(X_stuttering_motion, X_stuttering_mfcc):
    if len(motion.shape) < 1 or len(mfcc.shape) < 2:  # Ensure valid shapes
        print(f"Skipping invalid file: Motion {motion.shape}, MFCC {mfcc.shape}")
        continue
    min_length = min(len(motion), mfcc.shape[0])

    motion = motion[:min_length].reshape(-1, 1)  # Reshape to (time_steps, 1)
    mfcc = mfcc[:min_length]  # Trim to min_length

    stacked_features = np.hstack((motion, mfcc))  # Correct stacking
    X_stuttering.append(stacked_features)

    # Expand labels to match sequence length
    y_stuttering.extend([0] * min_length)  # Label 0 for each time step

for motion, mfcc in zip(X_lisp_motion, X_lisp_mfcc):
    if len(motion.shape) < 1 or len(mfcc.shape) < 2:  # Ensure valid shapes
        print(f"Skipping invalid file: Motion {motion.shape}, MFCC {mfcc.shape}")
        continue
    min_length = min(len(motion), mfcc.shape[0])

    motion = motion[:min_length].reshape(-1, 1)  # Reshape to (time_steps, 1)
    mfcc = mfcc[:min_length]  # Trim to min_length

    stacked_features = np.hstack((motion, mfcc))  # Correct stacking
    X_lisp.append(stacked_features)

    # Expand labels to match sequence length
    y_lisp.extend([1] * min_length)  # Label 1 for each time step

# Ensure there's data before training
if not X_stuttering and not X_lisp:
    raise ValueError("No training data found! Ensure feature extraction scripts have been run.")

# Stack both categories into training data
X_train = np.vstack(X_stuttering + X_lisp)
y_train = np.array(y_stuttering + y_lisp)  # Convert to NumPy array

# Debugging Output
print(f"Final X_train shape: {X_train.shape}")  # Should be (total_samples, time_steps)
print(f"Final y_train shape: {y_train.shape}")  # Should match X_train (total_samples,)

# Ensure the shapes match
assert X_train.shape[0] == y_train.shape[0], "Error: X_train and y_train must have the same number of samples!"

# Ensure X_train is 3D for LSTM
X_train = np.array(X_train, dtype=np.float32)

# Final reshape
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # Reshape for LSTM

print(f"X_train shape after reshaping: {X_train.shape}")  # Should be (samples, time_steps, features)

# Define LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (stuttering or lisp)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train Model with Validation Split (20%)
history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)

# Save Model
model.save("data/model/speech_dysfunction_model.h5")

# Evaluate the Model
final_loss, final_accuracy = model.evaluate(X_train, y_train)
print(f"Final Training Accuracy: {final_accuracy:.4f}")

if __name__ == "__main__":
    # Train Model only if this script is executed directly
    history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)

print("Model training complete and saved.")