#Accessing files from our Google Drive. 
#Because we want to run the code in Google Colab. 
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os # For creating directories if needed
import glob # For listing files
import soundfile as sf # Added for writing audio files
import time # Added for timing inference

# --- Configuration Parameters ---
SR = 16000  # Sample rate (Hz)
FRAME_LENGTH = 512  # STFT window size (samples)
HOP_LENGTH = 128    # STFT hop size (samples)
N_FFT = FRAME_LENGTH # Number of FFT points, usually same as FRAME_LENGTH
INPUT_DIM = N_FFT // 2 + 1 # Dimension of magnitude spectrogram frame (N_FFT/2 + 1)

# FF-DNN Specific Parameters (Lightweight)
FFDNN_HIDDEN_LAYERS = 3 # Number of hidden layers
FFDNN_NEURONS_PER_LAYER = 128 # Number of neurons in each hidden layer
FFDNN_DROPOUT_RATE = 0.2 # Dropout rate for regularization

# Training Parameters
BATCH_SIZE = 64
EPOCHS = 50 # Reduced for demonstration; actual training might need more
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Noise Generation Parameters
SNRS_DB = [-5, 0, 5, 10] # Signal-to-Noise Ratios to generate noisy speech

# --- Helper Functions ---

def load_and_mix_audio(clean_audio_path, noise_type='white', snr_db=0, sr=SR):
    """
    Loads a clean audio file and mixes it with generated white noise at a specified SNR.

    Args:
        clean_audio_path (str): Path to the clean speech audio file.
        noise_type (str): Type of noise to generate. Currently only 'white' is supported.
        snr_db (int): Desired Signal-to-Noise Ratio in dB.
        sr (int): Sample rate for loading and mixing.

    Returns:
        tuple: (clean_audio, noisy_audio) as numpy arrays.
    """
    try:
        clean_audio, _ = librosa.load(clean_audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading clean audio {clean_audio_path}: {e}")
        return None, None

    if noise_type == 'white':
        # Generate white noise with the same length as clean_audio
        noise = np.random.randn(len(clean_audio))
    else:
        raise ValueError("Unsupported noise type. Only 'white' is supported for this methodology.")

    # Calculate power of clean speech and noise
    P_clean = np.sum(clean_audio**2) + 1e-10 # Add epsilon to prevent division by zero
    P_noise = np.sum(noise**2) + 1e-10 # Add epsilon to prevent division by zero

    # Calculate scaling factor for noise to achieve desired SNR
    # SNR_dB = 10 * log10(P_clean / P_noise_scaled)
    # P_noise_scaled = P_clean / (10^(SNR_dB / 10))
    noise_scaling_factor = np.sqrt(P_clean / (P_noise * (10**(snr_db / 10))))
    scaled_noise = noise * noise_scaling_factor

    noisy_audio = clean_audio + scaled_noise

    # Normalize noisy audio to prevent clipping, while maintaining relative levels
    max_amp = np.max(np.abs(noisy_audio))
    if max_amp > 1.0:
        noisy_audio = noisy_audio / max_amp
        clean_audio = clean_audio / max_amp # Scale clean audio proportionally

    return clean_audio, noisy_audio

def extract_features(audio, sr=SR, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, n_fft=N_FFT):
    """
    Extracts magnitude spectrogram and preserves phase from an audio signal.

    Args:
        audio (np.ndarray): Audio waveform.
        sr (int): Sample rate.
        frame_length (int): STFT window size.
        hop_length (int): STFT hop size.
        n_fft (int): Number of FFT points.

    Returns:
        tuple: (magnitude_spectrogram_log, phase_spectrogram)
    """
    stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=frame_length)
    magnitude_spectrogram = np.abs(stft_result)
    phase_spectrogram = np.angle(stft_result) # Preserve phase

    # Apply log transformation for better neural network performance
    # Add a small epsilon to avoid log(0)
    magnitude_spectrogram_log = np.log1p(magnitude_spectrogram)

    return magnitude_spectrogram_log, phase_spectrogram

def reconstruct_audio(enhanced_magnitude_log, phase_spectrogram, sr=SR, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, n_fft=N_FFT):
    """
    Reconstructs audio waveform from enhanced log-magnitude and original phase.

    Args:
        enhanced_magnitude_log (np.ndarray): Enhanced log-magnitude spectrogram.
        phase_spectrogram (np.ndarray): Original phase spectrogram (from noisy input).
        sr (int): Sample rate.
        frame_length (int): STFT window size.
        hop_length (int): STFT hop size.
        n_fft (int): Number of FFT points.

    Returns:
        np.ndarray: Reconstructed audio waveform.
    """
    # Inverse log transformation
    enhanced_magnitude = np.expm1(enhanced_magnitude_log)

    # Reconstruct complex spectrogram
    complex_spectrogram = enhanced_magnitude * np.exp(1j * phase_spectrogram)

    # Inverse STFT
    reconstructed_audio = librosa.istft(complex_spectrogram, hop_length=hop_length, win_length=frame_length)

    return reconstructed_audio

def build_ffdnn_model(input_dim, output_dim, hidden_layers, neurons_per_layer, dropout_rate):
    """
    Builds a Feed-Forward Deep Neural Network (FF-DNN) model.

    Args:
        input_dim (int): Dimension of the input feature (e.g., magnitude spectrogram frame size * context_window_size).
        output_dim (int): Dimension of the output (e.g., magnitude spectrogram frame size).
        hidden_layers (int): Number of hidden layers.
        neurons_per_layer (int): Number of neurons in each hidden layer.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        tf.keras.Model: Compiled Keras FF-DNN model.
    """
    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=(input_dim,)))

    # Hidden layers
    for _ in range(hidden_layers):
        model.add(layers.Dense(neurons_per_layer, activation='relu'))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Output layer (predicts clean magnitude spectrogram)
    model.add(layers.Dense(output_dim, activation='linear')) # Linear activation for regression

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse') # Using MSE as loss

    return model

def calculate_snr(clean_signal, noisy_signal):
    """
    Calculates the Signal-to-Noise Ratio (SNR) in dB.
    """
    # Ensure signals are non-zero to avoid division by zero
    clean_signal_power = np.sum(clean_signal**2) + 1e-10
    noise_power = np.sum((noisy_signal - clean_signal)**2) + 1e-10
    snr = 10 * np.log10(clean_signal_power / noise_power)
    return snr

def calculate_delta_snr(clean_signal, noisy_signal, enhanced_signal):
    """
    Calculates the SNR Improvement (Delta SNR) in dB.
    """
    input_snr = calculate_snr(clean_signal, noisy_signal)
    output_snr = calculate_snr(clean_signal, enhanced_signal)
    return output_snr - input_snr

# --- Main Demonstration Script ---

if __name__ == "__main__":
    print("--- Starting FF-DNN Methodology Demonstration ---")

    # --- 1. Dataset Preparation ---
    google_drive_audio_path = '/content/drive/MyDrive/VCTK Corpus (version 0.92)/p225/p225_002.wav' # <--- CHANGE THIS PATH

    clean_audio_paths = []
 
    # Check if the Google Drive path exists (only if drive is mounted)
     try:
        # This will only work if running in Colab and drive is mounted
        if os.path.exists('/content/drive'):
            if os.path.exists(google_drive_audio_path):
                clean_audio_paths.append(google_drive_audio_path)
                print(f"Using audio file from Google Drive: {google_drive_audio_path}")
            else:
                print(f"Google Drive file not found at {google_drive_audio_path}.")
        else:
            print("Google Drive not mounted. Skipping Google Drive file check.")
    except NameError:
        print("Not running in Google Colab or 'google.colab' not imported. Skipping Google Drive file check.")

    # If you have a directory of clean speech files in Google Drive:
    # clean_audio_dir_gdrive = '/content/drive/MyDrive/VCTK Corpus (version 0.92)/p225'
    # if os.path.isdir(clean_audio_dir_gdrive):
    #     gdrive_files = glob.glob(os.path.join(clean_audio_dir_gdrive, '*.wav'))
    #     if gdrive_files:
    #         clean_audio_paths = gdrive_files # Overwrite if files found
    #         print(f"Using {len(clean_audio_paths)} audio files from Google Drive directory.")
    #     else:
    #         print(f"No .wav files found in '{clean_audio_dir_gdrive}'.")
    # else:
    #     print(f"Google Drive directory not found at {clean_audio_dir_gdrive}.")


    all_noisy_mags_log = []
    all_clean_mags_log = []
    all_noisy_phases = [] # Store phases for reconstruction

    print(f"Preparing dataset from {len(clean_audio_paths)} clean audio files...")
    for i, clean_path in enumerate(clean_audio_paths):
        print(f"Processing {i+1}/{len(clean_audio_paths)}: {os.path.basename(clean_path)}")
        for snr_db in SNRS_DB:
            clean_audio, noisy_audio = load_and_mix_audio(clean_path, snr_db=snr_db)
            if clean_audio is None:
                continue # Skip if loading failed

            noisy_mag_log, noisy_phase = extract_features(noisy_audio)
            clean_mag_log, _ = extract_features(clean_audio)

            # Ensure consistent length for features, trim if necessary
            min_frames = min(noisy_mag_log.shape[1], clean_mag_log.shape[1])
            noisy_mag_log = noisy_mag_log[:, :min_frames]
            clean_mag_log = clean_mag_log[:, :min_frames]
            noisy_phase = noisy_phase[:, :min_frames] # Keep phase consistent

            all_noisy_mags_log.append(noisy_mag_log.T) # Transpose for (frames, bins)
            all_clean_mags_log.append(clean_mag_log.T) # Transpose for (frames, bins)
            all_noisy_phases.append(noisy_phase) # Keep original phase structure for reconstruction

    # Concatenate all data
    X = np.vstack(all_noisy_mags_log)
    y = np.vstack(all_clean_mags_log)

    print(f"Total frames for training: {X.shape[0]}. Feature dimension: {X.shape[1]}")

    # Ensure input and output dimensions match
    if X.shape[1] != INPUT_DIM or y.shape[1] != INPUT_DIM:
        print(f"Error: Feature dimensions mismatch. X: {X.shape}, y: {y.shape}, Expected INPUT_DIM: {INPUT_DIM}")
        exit()

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=42)
    print(f"Data split: Train {X_train.shape[0]} frames, Validation {X_val.shape[0]} frames.")

    # --- 2. Lightweight Neural Network Architecture (FF-DNN) ---
    ffdnn_model = build_ffdnn_model(
        input_dim=INPUT_DIM,
        output_dim=INPUT_DIM,
        hidden_layers=FFDNN_HIDDEN_LAYERS,
        neurons_per_layer=FFDNN_NEURONS_PER_LAYER,
        dropout_rate=FFDNN_DROPOUT_RATE
    )
    ffdnn_model.summary()

    # --- 3. Training Procedure ---
    print("\n--- Starting FF-DNN Training ---")

    # Using EarlyStopping callback to stop training when validation loss stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10, # Number of epochs with no improvement after which training will be stopped.
        restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
    )

    history = ffdnn_model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1 # Set to 0 for silent training
    )
    print("--- FF-DNN Training Finished ---")

    # --- 4. Speech Reconstruction ---
    print("\n--- Demonstrating Speech Reconstruction on a single example ---")

    # For demonstration, let's pick one of the generated noisy audios for inference
    # In a real scenario, you'd use a separate test set.

    # Re-generate one noisy example for clear demonstration of inference
    # Use the first path from your list
    clean_audio_test_example, noisy_audio_test_example = load_and_mix_audio(clean_audio_paths[0], snr_db=0)
    if clean_audio_test_example is None:
        print("Failed to load test example audio. Exiting demonstration.")
        exit()

    test_noisy_mag_log, test_noisy_phase = extract_features(noisy_audio_test_example)

    # Predict enhanced magnitude spectrogram
    # For inference time, we'll predict on a small batch or single frame
    # Ensure input is (batch_size, features)
    single_frame_input = test_noisy_mag_log.T[0:1, :] # Take the first frame as a batch of 1

    start_time = time.time()
    _ = ffdnn_model.predict(single_frame_input, verbose=0) # Warm-up run
    end_time = time.time()

    start_time = time.time()
    # Predict on the entire test spectrogram (transposed)
    enhanced_mag_log_frames = ffdnn_model.predict(test_noisy_mag_log.T, verbose=0)
    end_time = time.time()

    # Calculate inference time per frame
    inference_time_total_ms = (end_time - start_time) * 1000 # Convert to milliseconds
    num_frames_predicted = test_noisy_mag_log.shape[1]
    inference_time_per_frame_ms = inference_time_total_ms / num_frames_predicted


    enhanced_mag_log = enhanced_mag_log_frames.T # Transpose back to (freq_bins, num_frames)

    # Ensure the shape matches for reconstruction
    if enhanced_mag_log.shape != test_noisy_mag_log.shape:
        print(f"Warning: Shape mismatch after prediction. Expected {test_noisy_mag_log.shape}, got {enhanced_mag_log.shape}")
        # Attempt to resize if possible, or handle error
        if enhanced_mag_log.shape[0] == test_noisy_mag_log.shape[0] and enhanced_mag_log.shape[1] > test_noisy_mag_log.shape[1]:
             # If predicted has more frames (e.g., due to padding in training), trim
            enhanced_mag_log = enhanced_mag_log[:, :test_noisy_mag_log.shape[1]]
        elif enhanced_mag_log.shape[0] == test_noisy_mag_log.shape[0] and enhanced_mag_log.shape[1] < test_noisy_mag_log.shape[1]:
            # If predicted has fewer frames, pad with zeros (less ideal but for demo)
            padding = test_noisy_mag_log.shape[1] - enhanced_mag_log.shape[1]
            enhanced_mag_log = np.pad(enhanced_mag_log, ((0,0), (0,padding)), mode='constant')
        else:
            print("Shape mismatch cannot be easily resolved for demonstration.")
            exit()


    enhanced_audio = reconstruct_audio(enhanced_mag_log, test_noisy_phase)

    # --- 5. Simple Evaluation (Visual & Audio) ---
    print("\n--- Simple Evaluation (Visual & Audio) ---")

    # Ensure audio lengths match for comparison (soundfile might pad/trim)
    min_len = min(len(clean_audio_test_example), len(noisy_audio_test_example), len(enhanced_audio))
    clean_audio_test_example = clean_audio_test_example[:min_len]
    noisy_audio_test_example = noisy_audio_test_example[:min_len]
    enhanced_audio = enhanced_audio[:min_len]

    # Save example audio files using soundfile
    try:
        sf.write('clean_example.wav', clean_audio_test_example, SR)
        sf.write('noisy_example.wav', noisy_audio_test_example, SR)
        sf.write('enhanced_example_ffdnn.wav', enhanced_audio, SR)
        print("Example audio files saved: clean_example.wav, noisy_example.wav, enhanced_example_ffdnn.wav")
    except Exception as e:
        print(f"Could not save audio files. Ensure 'soundfile' is installed: pip install soundfile. Error: {e}")

    # --- Calculate and Print Metrics ---
    # It's important that the clean, noisy, and enhanced signals are aligned and have the same sample rate.
  
    # Calculate SNR Improvement
    delta_snr = calculate_delta_snr(clean_audio_test_example, noisy_audio_test_example, enhanced_audio)
    print(f"SNR Improvement (Delta SNR): {delta_snr:.2f} dB")

    # --- Calculate and Print Model Complexity/Efficiency Metrics ---
    total_params = ffdnn_model.count_params()
    print(f"Trainable Parameters: {total_params}")
    print(f"Inference Time per Frame: {inference_time_per_frame_ms:.4f} ms/frame")

    # Plot spectrograms for visual comparison
    plt.figure(figsize=(15, 9))

    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(np.expm1(clean_mag_log), ref=np.max),
                             sr=SR, hop_length=HOP_LENGTH, y_axis='log', x_axis='time')
    plt.title('Clean Speech Spectrogram (Log-Magnitude)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.expm1(test_noisy_mag_log), ref=np.max),
                             sr=SR, hop_length=HOP_LENGTH, y_axis='log', x_axis='time')
    plt.title('Noisy Speech Spectrogram (Log-Magnitude)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(np.expm1(enhanced_mag_log), ref=np.max),
                             sr=SR, hop_length=HOP_LENGTH, y_axis='log', x_axis='time')
    plt.title('Enhanced Speech Spectrogram (FF-DNN Log-Magnitude)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    plt.show()

    print("\n--- FF-DNN Methodology Demonstration Complete ---")
