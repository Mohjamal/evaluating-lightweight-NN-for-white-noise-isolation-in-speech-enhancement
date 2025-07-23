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

# Small CNN Specific Parameters
CNN_FILTERS = 32 # Number of filters in convolutional layers
CNN_KERNEL_SIZE = (3, 1) # Kernel size (freq_bins, time_steps) - 1 for no temporal context
CNN_POOL_SIZE = (2, 1) # Pooling size (freq_bins, time_steps)
CNN_HIDDEN_DENSE_NEURONS = 128 # Neurons in dense layer after flattening
CNN_DROPOUT_RATE = 0.2 # Dropout rate for regularization

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

def build_small_cnn_model(input_dim, output_dim, cnn_filters, cnn_kernel_size, cnn_pool_size, cnn_hidden_dense_neurons, cnn_dropout_rate):
    """
    Builds a Small Convolutional Neural Network (CNN) model.

    Args:
        input_dim (int): Dimension of the input feature (e.g., magnitude spectrogram frame size).
        output_dim (int): Dimension of the output (e.g., magnitude spectrogram frame size).
        cnn_filters (int): Number of filters for convolutional layers.
        cnn_kernel_size (tuple): Kernel size for Conv2D layers (height, width).
        cnn_pool_size (tuple): Pool size for MaxPooling2D layers (height, width).
        cnn_hidden_dense_neurons (int): Number of neurons in the dense layer after flattening.
        cnn_dropout_rate (float): Dropout rate for regularization.

    Returns:
        tf.keras.Model: Compiled Keras Small CNN model.
    """
    model = models.Sequential()

    # Input layer: (freq_bins, 1 time step, 1 channel)
    model.add(layers.Input(shape=(input_dim, 1, 1)))

    # First Convolutional Block
    model.add(layers.Conv2D(cnn_filters, cnn_kernel_size, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=cnn_pool_size, padding='same'))
    if cnn_dropout_rate > 0:
        model.add(layers.Dropout(cnn_dropout_rate))

    # Second Convolutional Block (optional, but good for depth)
    model.add(layers.Conv2D(cnn_filters * 2, cnn_kernel_size, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=cnn_pool_size, padding='same'))
    if cnn_dropout_rate > 0:
        model.add(layers.Dropout(cnn_dropout_rate))

    # Flatten the output of the convolutional layers
    model.add(layers.Flatten())

    # Dense layer
    model.add(layers.Dense(cnn_hidden_dense_neurons, activation='relu'))
    if cnn_dropout_rate > 0:
        model.add(layers.Dropout(cnn_dropout_rate))

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
    print("--- Starting Small CNN Methodology Demonstration ---")

    # --- 1. Dataset Preparation ---
    google_drive_audio_path = '/content/drive/MyDrive/VCTK Corpus (version 0.92)/p225/p225_002.wav'

    clean_audio_paths = []
    # Check if the Google Drive path exists (only if drive is mounted)
    # The 'google.colab' module is only available in Colab.
    # We use a try-except block to safely check if drive is mounted.
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
    except NameError: # This happens if 'google.colab' was never imported (e.g., not in Colab)
        print("Not running in Google Colab or 'google.colab' not imported. Skipping Google Drive file check.")

    # If you have a directory of clean speech files in Google Drive:
    # clean_audio_dir_gdrive = '/content/drive/My Drive/YourCleanSpeechFolder/' # <--- CHANGE THIS PATH
    # if os.path.isdir(clean_audio_dir_gdrive):
    #     gdrive_files = glob.glob(os.path.join(clean_audio_dir_gdrive, '*.wav'))
    #     if gdrive_files:
    #         clean_audio_paths = gdrive_files # Overwrite if files found
    #         print(f"Using {len(gdrive_files)} audio files from Google Drive directory.")
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
    X_flat = np.vstack(all_noisy_mags_log) # (total_frames, freq_bins)
    y_flat = np.vstack(all_clean_mags_log) # (total_frames, freq_bins)

    # Reshape X for CNN input: (num_samples, freq_bins, 1, 1)
    X_cnn_input = X_flat.reshape(X_flat.shape[0], X_flat.shape[1], 1, 1)
    y_cnn_output = y_flat # y remains (num_samples, freq_bins)

    print(f"Total frames for training: {X_cnn_input.shape[0]}. Input shape for CNN: {X_cnn_input.shape[1:]}")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_cnn_input, y_cnn_output, test_size=VALIDATION_SPLIT, random_state=42)
    print(f"Data split: Train {X_train.shape[0]} frames, Validation {X_val.shape[0]} frames.")

    # --- 2. Lightweight Neural Network Architecture (Small CNN) ---
    small_cnn_model = build_small_cnn_model(
        input_dim=INPUT_DIM,
        output_dim=INPUT_DIM,
        cnn_filters=CNN_FILTERS,
        cnn_kernel_size=CNN_KERNEL_SIZE,
        cnn_pool_size=CNN_POOL_SIZE,
        cnn_hidden_dense_neurons=CNN_HIDDEN_DENSE_NEURONS,
        cnn_dropout_rate=CNN_DROPOUT_RATE
    )
    small_cnn_model.summary()

    # --- 3. Training Procedure ---
    print("\n--- Starting Small CNN Training ---")

    # Using EarlyStopping callback to stop training when validation loss stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10, # Number of epochs with no improvement after which training will be stopped.
        restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
    )

    history = small_cnn_model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1 # Set to 0 for silent training
    )
    print("--- Small CNN Training Finished ---")

    # Plot training history (optional)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Small CNN Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

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

    test_noisy_mag_log, test_noisy_phase = extract_features(noisy_audio_test_example) # (freq_bins, num_frames)

    # Reshape test input for CNN: (num_frames, freq_bins, 1, 1)
    test_cnn_input = test_noisy_mag_log.T.reshape(test_noisy_mag_log.shape[1], test_noisy_mag_log.shape[0], 1, 1)

    # Predict enhanced magnitude spectrogram
    # For inference time, we'll predict on a small batch or single frame
    single_frame_input = test_cnn_input[0:1, :, :, :] # Take the first frame as a batch of 1

    start_time = time.time()
    _ = small_cnn_model.predict(single_frame_input, verbose=0) # Warm-up run
    end_time = time.time()

    start_time = time.time()
    # Predict on the entire test spectrogram (reshaped for CNN)
    enhanced_mag_log_frames = small_cnn_model.predict(test_cnn_input, verbose=0) # Output is (num_frames, INPUT_DIM)
    end_time = time.time()

    # Calculate inference time per frame
    inference_time_total_ms = (end_time - start_time) * 1000 # Convert to milliseconds
    num_frames_predicted = test_cnn_input.shape[0]
    inference_time_per_frame_ms = inference_time_total_ms / num_frames_predicted

    # Transpose back to (freq_bins, num_frames) for reconstruction
    enhanced_mag_log = enhanced_mag_log_frames.T

    # Ensure the shape matches for reconstruction (phase_spectrogram is (freq_bins, num_frames))
    # The number of frames in enhanced_mag_log might be slightly different due to padding/trimming in STFT/ISTFT
    # Ensure they have the same number of frames for audio reconstruction.
    min_frames_for_reconstruction = min(enhanced_mag_log.shape[1], test_noisy_phase.shape[1])
    enhanced_mag_log = enhanced_mag_log[:, :min_frames_for_reconstruction]
    test_noisy_phase_trimmed = test_noisy_phase[:, :min_frames_for_reconstruction]


    enhanced_audio = reconstruct_audio(enhanced_mag_log, test_noisy_phase_trimmed)

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
        sf.write('enhanced_example_small_cnn.wav', enhanced_audio, SR)
        print("Example audio files saved: clean_example.wav, noisy_example.wav, enhanced_example_small_cnn.wav")
    except Exception as e:
        print(f"Could not save audio files. Ensure 'soundfile' is installed: pip install soundfile. Error: {e}")

    # --- Calculate and Print Metrics ---
    # It's important that the clean, noisy, and enhanced signals are aligned and have the same sample rate.

    # Calculate SNR Improvement
    delta_snr = calculate_delta_snr(clean_audio_test_example, noisy_audio_test_example, enhanced_audio)
    print(f"SNR Improvement (Delta SNR): {delta_snr:.2f} dB")

    # --- Calculate and Print Model Complexity/Efficiency Metrics ---
    total_params = small_cnn_model.count_params()
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
    plt.title('Enhanced Speech Spectrogram (Small CNN Log-Magnitude)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    plt.show()

    print("\n--- Small CNN Methodology Demonstration Complete ---")