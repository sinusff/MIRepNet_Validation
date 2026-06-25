
# Data-Flow Map (Step 1.1)
**Author:** Sina Feyzi

## How Data Flows in MIRepNet
The pipeline starts from `finetune.py` which calls `run_experiment` (in `utils/utils.py`). 
This function uses `EEGDataset` (in `dataset.py`) to load the `.npy` files and feed them into the model.

### Answers to the Project Guide questions:
**(a) Expected Array Shape:**
Based on `dataset.py`, for the BNCI2014004 dataset, the code truncates the time series to 1000 time points. The channels must match the predefined template. Therefore, the expected shape of the input array `X.npy` is:
`[Number of Trials, Number of Template Channels, 1000]`

**(b) Channel Template and Euclidean Alignment:**
The Euclidean Alignment and spatial interpolation onto the `utils/channel_list.py` template are **NOT** applied at load time inside the provided scripts. The model expects these transformations to be **already baked into the `.npy` files** before training begins.
