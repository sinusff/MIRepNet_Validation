import os
import mne
from moabb.datasets import BNCI2014_004
from moabb.paradigms import LeftRightImagery

# Output directory set directly to 'data'
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

print("Converting local BNCI2014004 .mat data to EEGLAB .set format inside 'data/' folder...\n")

dataset = BNCI2014_004()
paradigm = LeftRightImagery()

for sub in range(1, 10):
    print(f"Processing Subject {sub}/9...")
    
    # Extract trials for current subject
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[sub])
    
    # Define channels (C3, Cz, C4) and sampling frequency (250 Hz)
    ch_names = ['C3', 'Cz', 'C4']
    ch_types = ['eeg'] * 3
    sfreq = 250.0
    
    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage('standard_1020')
    
    # Create MNE EpochsArray object
    epochs = mne.EpochsArray(X, info)
    
    # Export directly to 'data/subject_01.set' ... 'data/subject_09.set'
    set_filename = os.path.join(output_dir, f'subject_{sub:02d}.set')
    mne.export.export_epochs(set_filename, epochs, fmt='eeglab', overwrite=True)
    print(f"   ✅ Saved: {set_filename}")

print("\n🎉 All 9 subjects successfully converted and saved inside the 'data/' folder!")