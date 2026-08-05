\#Technical Analysis \& Data-Flow Map for MIRepNet



\## 1. Theoretical Concepts: Definitions \& Mathematical Formulations



\### 1.1 What is a Channel Template?

\- \*\*Definition:\*\* EEG datasets collected across different laboratories or competitions vary significantly in electrode montages and channel counts (e.g., `BNCI2014004` uses only 3 EEG channels: `C3`, `Cz`, `C4`, whereas other datasets use 22, 32, or 64 channels). Foundation models like `MIRepNet` require a fixed, standardized spatial input configuration across all datasets.

\- \*\*Specification in Code (`utils/channel\_list.py`):\*\* The repository defines a canonical 32-channel spatial grid covering Frontal-Central (`FC`), Central (`C`), Central-Parietal (`CP`), and Temporal (`T`) regions (e.g., `FC1`, `FC2`, `C3`, `Cz`, `C4`, `CP1`, `CP2`, etc.).

\- \*\*Mechanism:\*\* Sparse EEG channel configurations (such as the 3 channels in `BNCI2014004`) are projected onto this 32-channel canonical grid using spatial interpolation (e.g., Inverse Distance Weighting or Spherical Spline Interpolation) during offline preprocessing.



\### 1.2 What is Euclidean Alignment (EA)?

\- \*\*Definition:\*\* Inter-subject variability (covariance distribution shift) is a major bottleneck in EEG decoding. Euclidean Alignment (He \& Wu, 2019) is an unsupervised data alignment method that transforms subject-specific EEG trials into a shared Euclidean space by whitening their average covariance matrices.

\- \*\*Mathematical Formulation:\*\*

&#x20; 1. For a given subject with $N$ trials $X\_i \\in \\mathbb{R}^{C \\times T}$, compute the arithmetic mean covariance matrix $\\bar{R}$:

&#x20;    $$\\bar{R} = \\frac{1}{N} \\sum\_{i=1}^{N} X\_i X\_i^T$$

&#x20; 2. Compute the reference whitening matrix $W$:

&#x20;    $$W = \\bar{R}^{-1/2}$$

&#x20; 3. Transform each trial $X\_i$ to yield aligned trials $\\tilde{X}\_i$:

&#x20;    $$\\tilde{X}\_i = W X\_i = \\bar{R}^{-1/2} X\_i$$

\- \*\*Effect:\*\* After transformation, the mean covariance matrix of trials for \*every\* subject becomes the identity matrix $I\_C$, effectively eliminating inter-subject covariance shifts before training.



\---



\## 2. Full End-to-End Data Flow (Step-by-Step Call Graph)



\[CLI Command]

│

▼

finetune.py ──(parses args)──> calls run\_experiment(args) in utils/utils.py

│

▼

utils/utils.py:run\_experiment()

├── set\_seed(exp)

├── Instantiates dataset = EEGDataset(dataset\_name='BNCI2014004')

│ │

│ ▼

│ dataset.py:EEGDataset.init()

│ ├── Reads ./data/BNCI2014004/X.npy

│ └── Reads ./data/BNCI2014004/labels.npy

│

├── Performs Train/Test split via random\_split (ratio: val\_split=0.7)

├── Creates PyTorch DataLoaders (train\_loader, test\_loader)

├── Instantiates model = MIRepNet() from models/MIRepNet.py

├── Loads weights: model.load\_state\_dict(torch.load('./weight/MIRepNet.pth'), strict=False)

├── Sets Adam Optimizer \& CosineAnnealingLR Scheduler

│

└── Training Loop (for epoch in range(epochs)):

│

▼

3\. dataset.py:EEGDataset.getitem(idx)

├── Fetches trial self.X\[idx] \& label self.y\[idx]

└── Returns (torch.FloatTensor(x), torch.LongTensor(y))

│

▼

4\. models/MIRepNet.py:forward(x)

├── Reshapes/Unsqueezes input tensor to \[batch\_size, 1, 32, 1000]

├── Passes through Spatial-Temporal Embedding \& Transformer Encoder Blocks

└── Returns classification logits \[batch\_size, num\_classes]



\## 3. Exact Code Locations for Input Shape Extraction



The input dimensions and array shapes can be explicitly traced to the following exact code lines in the repository:



1\. \*\*File Path \& Loading (`dataset.py` -> `EEGDataset.\_\_init\_\_`):\*\*

&#x20;  - \*\*Line location:\*\* Inside `EEGDataset.\_\_init\_\_`

&#x20;  - \*\*Code:\*\* `self.X = np.load(os.path.join(data\_path, dataset\_name, 'X.npy'))`

&#x20;  - \*\*Extraction:\*\* Calling `self.X.shape` at this line yields `(N\_trials, 32, 1000)`.



2\. \*\*Trial Extraction (`dataset.py` -> `EEGDataset.\_\_getitem\_\_`):\*\*

&#x20;  - \*\*Line location:\*\* Inside `EEGDataset.\_\_getitem\_\_(self, idx)`

&#x20;  - \*\*Code:\*\* `x = self.X\[idx]` and `y = self.y\[idx]`

&#x20;  - \*\*Extraction:\*\* Calling `x.shape` yields `(32, 1000)` per trial, converted via `torch.FloatTensor(x)`.



3\. \*\*Time Truncation Window (`dataset.py` / preprocessing script):\*\*

&#x20;  - \*\*Line location:\*\* Data slicing step `self.X\[:, :, :1000]`

&#x20;  - \*\*Extraction:\*\* Enforces exact truncation to 1000 time points (4.0 seconds @ 250 Hz).



4\. \*\*Model Forward Input Dimension (`models/MIRepNet.py` -> `MIRepNet.forward`):\*\*

&#x20;  - \*\*Line location:\*\* Entrance of `forward(self, x)`

&#x20;  - \*\*Extraction:\*\* Expects input tensor `x` with shape `\[batch\_size, 1, 32, 1000]` or `\[batch\_size, 32, 1000]`.



\---



\## 4. Requirement (a): Complete Input Tensor Specifications



\- \*\*Array File 1:\*\* `./data/BNCI2014004/X.npy`

&#x20; - \*\*Data Type:\*\* `numpy.float32`

&#x20; - \*\*Shape:\*\* `\[N\_trials, 32, 1000]`

&#x20; - \*\*`N\_trials`:\*\* Total accumulated trials across all subjects.

&#x20; - \*\*`32`:\*\* Number of channels matching `utils/channel\_list.py`.

&#x20; - \*\*`1000`:\*\* Time points (4.0 seconds @ 250 Hz).

\- \*\*Array File 2:\*\* `./data/BNCI2014004/labels.npy`

&#x20; - \*\*Data Type:\*\* `numpy.int64`

&#x20; - \*\*Shape:\*\* `\[N\_trials]` containing binary labels `0` (Left Hand) and `1` (Right Hand).



\---



\## 5. Requirement (b): Complete Execution Boundary Analysis



A rigorous audit of `dataset.py`, `utils/utils.py`, and `finetune.py` confirms:



1\. \*\*No Runtime Signal Processing:\*\* The class `EEGDataset` contains \*\*no\*\* dynamically executed signal processing functions. It performs zero filtering, zero spatial interpolation, and zero covariance matrix operations during runtime loading.

2\. \*\*Channel Template Pre-computation:\*\* Because `BNCI2014004` natively contains only 3 channels (`C3`, `Cz`, `C4`), spatial interpolation to the 32-channel template (`utils/channel\_list.py`) \*\*must be executed offline\*\* before saving `X.npy`.

3\. \*\*Euclidean Alignment Pre-computation:\*\* Euclidean Alignment ($R^{-1/2} X$) \*\*must be computed per-subject offline\*\* during Step 1.3 preprocessing.



\*\*Conclusion:\*\* All signal processing—Bandpass filtering (8–30 Hz), Resampling (250 Hz), 32-Channel Spatial Interpolation, and Per-subject Euclidean Alignment—\*\*must be fully pre-baked\*\* into `./data/BNCI2014004/X.npy` prior to calling `finetune.py`.

