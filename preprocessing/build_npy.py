import os
import mne
import numpy as np
from scipy.signal import butter, filtfilt
from moabb.datasets import BNCI2014_004
from moabb.paradigms import LeftRightImagery


def bandpass_filter(data, lowcut=8.0, highcut=30.0, fs=250.0, order=4):
    """Apply 4th-order Butterworth bandpass filter (8-30 Hz) per MIRepNet paper."""
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


def main():
    # 1. Point MNE to the user's exact downloaded local .mat directory
    local_data_dir = r'C:\Users\ASUS\MIRepNet_Validation\data\mne_data'
    mne.set_config('MNE_DATASETS_BNCI_PATH', local_data_dir)
    
    output_dir = './data/BNCI2014004'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading local .mat files from: {local_data_dir}")
    print("Building X.npy and labels.npy matching author's dataset.py hardcoded indices...\n")

    dataset = BNCI2014_004()
    paradigm = LeftRightImagery()

    # Exact array size required by dataset.py hardcoded indices (6360 trials)
    total_array_length = 6360
    n_channels = 3      # C3, Cz, C4
    n_timepoints = 1000 # 1000 timepoints (4s @ 250Hz)

    X_full = np.zeros((total_array_length, n_channels, n_timepoints), dtype=np.float32)
    labels_full = np.zeros((total_array_length,), dtype=np.int64)

    # Exact hardcoded index map from author's dataset.py
    subject_indices = {
        0: (400, 560),     # Subject 1 (160 trials)
        1: (1120, 1240),   # Subject 2 (120 trials)
        2: (1800, 1960),   # Subject 3 (160 trials)
        3: (2540, 2700),   # Subject 4 (160 trials)
        4: (3280, 3440),   # Subject 5 (160 trials)
        5: (4000, 4160),   # Subject 6 (160 trials)
        6: (4720, 4880),   # Subject 7 (160 trials)
        7: (5480, 5640),   # Subject 8 (160 trials)
        8: (6200, 6360)    # Subject 9 (160 trials)
    }

    for sub_idx in range(9):
        sub_id = sub_idx + 1
        start_idx, end_idx = subject_indices[sub_idx]
        expected_trials = end_idx - start_idx

        print(f"--> Processing Subject {sub_id}/9 from local disk (Target Slice: [{start_idx}:{end_idx}])...")

        try:
            # Load local downloaded data via MOABB
            X_sub, labels_sub, _ = paradigm.get_data(dataset=dataset, subjects=[sub_id])
            
            # Apply 8-30 Hz bandpass filter
            X_sub_filtered = bandpass_filter(X_sub, lowcut=8.0, highcut=30.0, fs=250.0)
            X_sub_truncated = X_sub_filtered[:, :, :1000]
            
            # Binary labels mapping ('left_hand' -> 0, 'right_hand' -> 1)
            binary_labels = np.array([0 if l == 'left_hand' else 1 for l in labels_sub], dtype=np.int64)
            
            # Place trials into exact hardcoded index slice
            actual_trials = min(len(X_sub_truncated), expected_trials)
            X_full[start_idx : start_idx + actual_trials] = X_sub_truncated[:actual_trials]
            labels_full[start_idx : start_idx + actual_trials] = binary_labels[:actual_trials]
            print(f"   ✅ Placed {actual_trials} trials into indices [{start_idx}:{start_idx + actual_trials}]")

        except Exception as e:
            print(f"   ❌ Error processing Subject {sub_id}: {e}")

    # Save final arrays to ./data/BNCI2014004/
    x_path = os.path.join(output_dir, 'X.npy')
    labels_path = os.path.join(output_dir, 'labels.npy')

    np.save(x_path, X_full)
    np.save(labels_path, labels_full)

    print("\n🎉 Step 1.3 Preprocessing completed successfully!")
    print(f"✅ X.npy saved at: {x_path} | Shape: {X_full.shape}")
    print(f"✅ labels.npy saved at: {labels_path} | Shape: {labels_full.shape}")


if __name__ == '__main__':
    main()