import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from moabb.datasets import BNCI2014_004
from moabb.paradigms import LeftRightImagery


def bandpass_filter(data, lowcut=8.0, highcut=14.0, fs=250.0, order=4):
    """
    Apply 4th-order Butterworth bandpass filter (8-14 Hz) for mu/alpha rhythm.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


def compute_erd_percent(epochs, fs=250.0, baseline_samples=250):
    """
    Compute percentage ERD/ERS (% change relative to pre-cue baseline):
    ERD% = [(Power - Baseline_Power) / Baseline_Power] * 100
    """
    # 1. Bandpass filter in 8-14 Hz
    filtered = bandpass_filter(epochs, lowcut=8.0, highcut=14.0, fs=fs)
    
    # 2. Instantaneous power via Hilbert transform envelope
    analytic_signal = hilbert(filtered, axis=-1)
    inst_power = np.abs(analytic_signal) ** 2
    
    # 3. Average power across trials
    mean_power = np.mean(inst_power, axis=0)  # Shape: [channels, time]
    
    # 4. Calculate baseline power (first 1.0 second before/at cue)
    baseline_power = np.mean(mean_power[:, :baseline_samples], axis=-1, keepdims=True)
    
    # 5. Calculate percentage ERD/ERS
    erd_percent = ((mean_power - baseline_power) / baseline_power) * 100.0
    return erd_percent


def main():
    os.makedirs('results', exist_ok=True)

    print("Loading downloaded BNCI2014004 dataset from local disk...")
    dataset = BNCI2014_004()
    paradigm = LeftRightImagery()

    # Load data for Subject 1, Subject 2, and Subject 3
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1, 2, 3])

    # Plot 2x2 grid comparing Left Hand vs Right Hand Imagery for Subject 1 & 2
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for idx, sub_id in enumerate([1, 2]):
        # --- Left Hand Imagery ---
        left_mask = (meta['subject'] == sub_id) & (labels == 'left_hand')
        left_epochs = X[left_mask]
        left_erd = compute_erd_percent(left_epochs)
        times = np.linspace(0, 4.0, left_erd.shape[1])

        ax_left = axes[idx, 0]
        ax_left.plot(times, left_erd[0], 'b-', label='C3 (Left Hem.)', linewidth=2)
        ax_left.plot(times, left_erd[2], 'r--', label='C4 (Right Hem. - Contralateral)', linewidth=2)
        ax_left.axhline(0, color='black', linestyle=':', alpha=0.7)
        ax_left.axvline(1.0, color='gray', linestyle='--', label='Cue Onset')
        ax_left.set_title(f'Subject {sub_id} - Left Hand Imagery', fontsize=13, fontweight='bold')
        ax_left.set_xlabel('Time (seconds)', fontsize=11)
        ax_left.set_ylabel('ERD / ERS (%)', fontsize=11)
        ax_left.legend(fontsize=10)
        ax_left.grid(True, alpha=0.3)

        # --- Right Hand Imagery ---
        right_mask = (meta['subject'] == sub_id) & (labels == 'right_hand')
        right_epochs = X[right_mask]
        right_erd = compute_erd_percent(right_epochs)

        ax_right = axes[idx, 1]
        ax_right.plot(times, right_erd[0], 'b-', label='C3 (Left Hem. - Contralateral)', linewidth=2)
        ax_right.plot(times, right_erd[2], 'r--', label='C4 (Right Hem.)', linewidth=2)
        ax_right.axhline(0, color='black', linestyle=':', alpha=0.7)
        ax_right.axvline(1.0, color='gray', linestyle='--', label='Cue Onset')
        ax_right.set_title(f'Subject {sub_id} - Right Hand Imagery', fontsize=13, fontweight='bold')
        ax_right.set_xlabel('Time (seconds)', fontsize=11)
        ax_right.set_ylabel('ERD / ERS (%)', fontsize=11)
        ax_right.legend(fontsize=10)
        ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/erd_plots.png', dpi=300)
    print("✅ Precise ERD/ERS percentage figure saved to 'results/erd_plots.png'.")


if __name__ == '__main__':
    main()