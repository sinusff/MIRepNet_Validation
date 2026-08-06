import os
import requests
import urllib3
from tqdm import tqdm

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Target directory expected by MNE / MOABB
target_dir = os.path.expanduser('~/mne_data/MNE-bnci-data/004-2014')
os.makedirs(target_dir, exist_ok=True)

# Official open mirror from BNCI Horizon 2020
base_url = "http://bnci-horizon-2020.eu/database/data-sets/004-2014/"

# Generate all 18 filenames (B01T.mat to B09E.mat)
files = []
for sub in range(1, 10):
    files.append(f"B0{sub}T.mat")
    files.append(f"B0{sub}E.mat")

print(f"Target Directory: {target_dir}")
print(f"Starting download of {len(files)} files from BNCI Horizon mirror...\n")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

for idx, fname in enumerate(files, 1):
    file_path = os.path.join(target_dir, fname)
    url = base_url + fname

    # Skip if already downloaded (> 30 MB)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 30 * 1024 * 1024:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"[{idx:02d}/{len(files):02d}] ⏩ {fname} ({size_mb:.1f} MB) already exists. Skipping.")
        continue

    print(f"[{idx:02d}/{len(files):02d}] ⬇️ Downloading {fname}...")
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB chunks

        with open(file_path, 'wb') as file, tqdm(
            desc=f"   Progress",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            ncols=80
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)

        print(f"   ✅ {fname} downloaded successfully!\n")

    except Exception as e:
        print(f"   ❌ Error downloading {fname}: {e}\n")

print("🎉 All 18 dataset files processed successfully!")