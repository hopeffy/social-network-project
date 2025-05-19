# Social Network Project

## Project Overview
This project implements a social network analysis system using temporal graph neural networks.

## Project Structure
```
.
├── data/                     # Data directory
│   ├── dataset_a.py         # Dataset loading and preprocessing scripts
│   ├── edges_train_A.csv    # Training edge data with timestamps
│   └── node_features.csv    # Node feature data
│
├── models/                   # Model implementations
│   └── temporal_gat.py      # Temporal Graph Attention Network implementation
│
├── training/                 # Training related code
│   └── trainer.py           # Trainer implementation with evaluation logic
│
├── test-data/               # Test datasets directory
│   └── input_A_initial.csv  # Initial test edge data
│
├── test_reports/            # Directory for test result reports
│
├── checkpoints/             # Model checkpoints directory
│   └── edges_train_A/       # Checkpoints for specific datasets
│
├── checkpoints_a5000/       # Alternative checkpoints directory
│
├── main.py                  # Main training script
├── test-model.py           # Standard model test script
├── test-model-2m.py        # 2M parameter model test script
├── test-model-27m.py       # 27M parameter model test script
```
