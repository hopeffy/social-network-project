# social-network-project

.
├── data/                    # Data directory
│   ├── dataset_a.py         # Dataset loading and preprocessing
│   ├── edges_train_A.csv    # Training edge data with timestamps
│   └── node_features.csv    # Node feature data
├── models/                  # Model implementations
│   └── temporal_gat.py      # Temporal Graph Attention Network
├── training/                # Training code
│   └── trainer.py           # Trainer implementation with evaluation
├── test-data/               # Test datasets
│   └── input_A_initial.csv  # Test edge data
├── test_reports/            # Test result reports
├── checkpoints/             # Model checkpoints
│   └── edges_train_A/       # Checkpoints for specific datasets
├── checkpoints_a5000/       # Alternative checkpoints directory
├── main.py                  # Main training script
├── test-model.py            # Standard model test script
├── test-model-2m.py         # 2M parameter model test script
├── test-model-27m.py        # 27M parameter model test script
├── environment.yml          # Conda environment specification
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation