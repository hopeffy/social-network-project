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

## Setup and Installation

### Dependencies
The project has two options for managing dependencies:

1. Using Conda (Recommended):
```bash
conda env create -f environment.yml
conda activate social-network
```

2. Using pip:
```bash
pip install -r requirements.txt
```

## Usage

### Training
To train the model, run:
```bash
python main.py
```

### Testing
There are three different test scripts available:

1. Standard test:
```bash
python test-model.py
```

2. 2M parameter model test:
```bash
python test-model-2m.py
```

3. 27M parameter model test:
```bash
python test-model-27m.py
```

## Project Files Description

### Core Files
- `main.py`: Main entry point for training the model
- `test-model.py`: Standard testing script
- `test-model-2m.py`: Testing script for 2M parameter model
- `test-model-27m.py`: Testing script for 27M parameter model

### Data Management
- `data/dataset_a.py`: Handles dataset loading and preprocessing
- `data/edges_train_A.csv`: Contains training edge data with timestamps
- `data/node_features.csv`: Contains node feature information

### Model Implementation
- `models/temporal_gat.py`: Implementation of Temporal Graph Attention Network

### Training
- `training/trainer.py`: Contains training logic and evaluation methods

### Testing
- `test-data/input_A_initial.csv`: Initial test data
- `test_reports/`: Directory containing test results and reports

### Model Checkpoints
- `checkpoints/`: Main checkpoints directory
- `checkpoints_a5000/`: Alternative checkpoints directory for specific configurations