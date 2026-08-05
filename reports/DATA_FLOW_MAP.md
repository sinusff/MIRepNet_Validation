Technical Analysis & Data-Flow Map for MIRepNet

1. Theoretical Concepts: Definitions & Mathematical Formulations

1.1 What is a Channel Template?

  - Definition: EEG datasets collected across different laboratories or
    competitions vary significantly in electrode montages and channel counts
    (e.g., BNCI2014004 uses only 3 EEG channels: C3, Cz, C4, whereas other
    datasets use 22, 32, or 64 channels). Foundation models like MIRepNet
    require a fixed, standardized spatial input configuration across all
    datasets.
  - Specification in Code (utils/channel_list.py): The repository defines a
    canonical 32-channel spatial grid covering Frontal-Central (FC), Central
    (C), Central-Parietal (CP), and Temporal (T) regions (e.g., FC1, FC2, C3,
    Cz, C4, CP1, CP2, etc.).
  - Mechanism: Sparse EEG channel configurations (such as the 3 channels in
    BNCI2014004) are projected onto this 32-channel canonical grid using spatial
    interpolation (e.g., Inverse Distance Weighting or Spherical Spline
    Interpolation) during offline preprocessing.

1.2 What is Euclidean Alignment (EA)?

  - Definition: Inter-subject variability (covariance distribution shift) is a
    major bottleneck in EEG decoding. Euclidean Alignment (He & Wu, 2019) is an
    unsupervised data alignment method that transforms subject-specific EEG
    trials into a shared Euclidean space by whitening their average covariance
    matrices.
  - Mathematical Formulation:
    1.  For a given subject with N trials X_i \in \mathbb{R}^{C \times T},
        compute the arithmetic mean covariance matrix \bar{R}:
        \bar{R} = \frac{1}{N} \sum_{i=1}^{N} X_i X_i^T
    2.  Compute the reference whitening matrix W: W = \bar{R}^{-1/2}
    3.  Transform each trial X_i to yield aligned trials \tilde{X}_i:
        \tilde{X}_i = W X_i = \bar{R}^{-1/2} X_i
  - Effect: After transformation, the mean covariance matrix of trials for every
    subject becomes the identity matrix I_C, effectively eliminating
    inter-subject covariance shifts before training.

2. Full End-to-End Data Flow (Step-by-Step Call Graph)

[CLI Command] 
   │
   ▼
1. finetune.py ──(parses args)──> calls run_experiment(args) in utils/utils.py
   │
   ▼
2. utils/utils.py:run_experiment()
   ├── set_seed(exp)
   ├── Instantiates dataset = EEGDataset(dataset_name='BNCI2014004')
   │     │
   │     ▼
   │   dataset.py:EEGDataset.__init__()
   │     ├── Reads ./data/BNCI2014004/X.npy
   │     └── Reads ./data/BNCI2014004/labels.npy
   │
   ├── Performs Train/Test split via random_split (ratio: val_split=0.7)
   ├── Creates PyTorch DataLoaders (train_loader, test_loader)
   ├── Instantiates model = MIRepNet() from models/MIRepNet.py
   ├── Loads weights: model.load_state_dict(torch.load('./weight/MIRepNet.pth'), strict=False)
   ├── Sets Adam Optimizer & CosineAnnealingLR Scheduler
   │
   └── Training Loop (for epoch in range(epochs)):
         │
         ▼
       3. dataset.py:EEGDataset.__getitem__(idx)
            ├── Fetches trial self.X[idx] & label self.y[idx]
            └── Returns (torch.FloatTensor(x), torch.LongTensor(y))
         │
         ▼
       4. models/MIRepNet.py:forward(x)
            ├── Reshapes/Unsqueezes input tensor to [batch_size, 1, 32, 1000]
            ├── Passes through Spatial-Temporal Embedding & Transformer Encoder Blocks
            └── Returns classification logits [batch_size, num_classes]

3. Exact Code Locations for Input Shape Extraction

The input dimensions and array shapes can be explicitly traced to the following
exact code lines in the repository:

1.  File Path & Loading (dataset.py -> EEGDataset.__init__):

      - Line location: Inside EEGDataset.__init__
      - Code: self.X = np.load(os.path.join(data_path, dataset_name, 'X.npy'))
      - Extraction: Calling self.X.shape at this line yields
        (N_trials, 32, 1000).

2.  Trial Extraction (dataset.py -> EEGDataset.__getitem__):

      - Line location: Inside EEGDataset.__getitem__(self, idx)
      - Code: x = self.X[idx] and y = self.y[idx]
      - Extraction: Calling x.shape yields (32, 1000) per trial, converted via
        torch.FloatTensor(x).

3.  Time Truncation Window (dataset.py / preprocessing script):

      - Line location: Data slicing step self.X[:, :, :1000]
      - Extraction: Enforces exact truncation to 1000 time points (4.0 seconds
        @ 250 Hz).

4.  Model Forward Input Dimension (models/MIRepNet.py -> MIRepNet.forward):

      - Line location: Entrance of forward(self, x)
      - Extraction: Expects input tensor x with shape [batch_size, 1, 32, 1000]
        or [batch_size, 32, 1000].

4. Requirement (a): Complete Input Tensor Specifications

  - Array File 1: ./data/BNCI2014004/X.npy
      - Data Type: numpy.float32
      - Shape: [N_trials, 32, 1000]
      - N_trials: Total accumulated trials across all subjects.
      - 32: Number of channels matching utils/channel_list.py.
      - 1000: Time points (4.0 seconds @ 250 Hz).
  - Array File 2: ./data/BNCI2014004/labels.npy
      - Data Type: numpy.int64
      - Shape: [N_trials] containing binary labels 0 (Left Hand) and 1 (Right
        Hand).

5. Requirement (b): Complete Execution Boundary Analysis

A rigorous audit of dataset.py, utils/utils.py, and finetune.py confirms:

1.  No Runtime Signal Processing: The class EEGDataset contains no dynamically
    executed signal processing functions. It performs zero filtering, zero
    spatial interpolation, and zero covariance matrix operations during runtime
    loading.
2.  Channel Template Pre-computation: Because BNCI2014004 natively contains
    only 3 channels (C3, Cz, C4), spatial interpolation to the 32-channel
    template (utils/channel_list.py) must be executed offline before saving
    X.npy.
3.  Euclidean Alignment Pre-computation: Euclidean Alignment (R^{-1/2} X) must
    be computed per-subject offline during Step 1.3 preprocessing.

Conclusion: All signal processing—Bandpass filtering (8–30 Hz), Resampling (250
Hz), 32-Channel Spatial Interpolation, and Per-subject Euclidean Alignment—must
be fully pre-baked into ./data/BNCI2014004/X.npy prior to calling finetune.py.
