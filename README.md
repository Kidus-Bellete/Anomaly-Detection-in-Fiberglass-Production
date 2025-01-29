# Fiberglass Anomaly Detection System

## Overview
An advanced anomaly detection system designed for identifying defects in fiberglass materials. The system combines EfficientNet-based feature extraction with an attention-enhanced FastFlow architecture to learn and detect anomalous patterns.

## Key Features
- Two-stage architecture combining feature extraction and normalizing flows
- Attention mechanisms for improved defect detection
- Robust validation and testing methodology
- Support for image preprocessing and augmentation
- Early stopping and model checkpointing
- Comprehensive metrics tracking

## Requirements
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.2
Pillow>=8.0.0
opencv-python>=4.5.0
tqdm>=4.50.0
```

## Project Structure
```
├── models/
│   ├── feature_extractor.py   # EfficientNet-based feature extraction
│   └── fastflow.py            # Attention-enhanced FastFlow implementation
├── utils/
│   ├── dataset.py             # Dataset handling and augmentation
│   ├── preprocessing.py       # Image preprocessing utilities
│   ├── visualization.py       # Visualization tools
│   └── metrics.py            # Evaluation metrics
├── train.py                   # Training script
├── test.py                    # Testing script
└── main.py                    # Main entry point
```

## Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd fiberglass-anomaly-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure
Organize your dataset as follows:
```
dataset/
├── train/
│   └── normal/          # Normal samples for training
└── test/
    ├── normal/          # Normal samples for testing
    └── anomalous/       # Anomalous samples for testing
```

## Configuration
Key configuration parameters in `main.py`:
```python
config = {
    'data_root': 'dataset',
    'checkpoint_dir': 'checkpoints',
    'output_dir': 'outputs',
    'efficientnet_version': 'efficientnet-b0',
    'hidden_dim': 64,
    'latent_dim': 16,
    'batch_size': 4,
    'learning_rate': 1e-4,
    'epochs': 200,
    'save_frequency': 10,
    'early_stopping_patience': 20,
    'preprocessing': True
}
```

## Usage

### Training
```bash
python main.py --mode train
```

### Testing
```bash
python main.py --mode test
```

### Training and Testing
```bash
python main.py --mode train_and_test
```

## Model Architecture

### Feature Extractor
- Based on EfficientNet-B0
- Pretrained on ImageNet
- Fine-tuned final layers
- Outputs high-dimensional feature representations

### FastFlow with Attention
- Spatial and channel attention mechanisms
- Normalizing flow architecture
- Multiple coupling layers
- Zero-centered target distribution

## Training Process
1. Feature extraction from normal samples
2. FastFlow training with attention mechanisms
3. Regular validation checks
4. Model checkpointing
5. Early stopping based on validation loss

## Results
The system outputs:
- Anomaly scores for test images
- ROC curves and AUC scores
- Precision-Recall curves
- Visualization of detected anomalies
- Training and validation loss curves

## Logging and Monitoring
- Comprehensive logging in `outputs/run.log`
- Training progress visualization
- Model checkpoints in `checkpoints/`
- Performance metrics tracking

## Best Practices
1. Use high-quality training data of normal samples
2. Enable preprocessing for enhanced defect detection
3. Adjust augmentation parameters based on your data
4. Monitor validation loss for optimal training
5. Use multiple forward passes during testing

## License


## Contributing


## Contact

