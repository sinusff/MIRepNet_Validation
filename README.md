# MIRepNet Validation & New Dataset Project
**Prepared by:** Sina Feizi  
**Supervisor:** Mohammad Nemati  

## Step 1.1 Environment Setup (Google Colab)
This project is set up and executed in **Google Colab** to leverage free GPUs and avoid package conflicts.

**GPU Check:** `True` (Using Google Colab T4 GPU)
**Pretrained Checkpoint:** `MIRepNet.pth` successfully downloaded and placed in `./weight/`.

### Setup Instructions for Colab:
```python
# 1. Install required packages
!pip install moabb huggingface_hub pyriemann braindecode matplotlib mne-bids

# 2. Download checkpoint
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="starself/MIRepNet", filename="MIRepNet.pth", local_dir="./weight")
