# main.py
import os
import argparse
import logging
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vetro.train import train
from vetro.test import test

def main():
    parser = argparse.ArgumentParser(description='Fiberglass Anomaly Detection')
    parser.add_argument('--mode', type=str, default='train',
                      choices=['train', 'test', 'train_and_test'],
                      help='Mode of operation')
    parser.add_argument('--config', type=str, default='config.py',
                      help='Path to config file')
    args = parser.parse_args()

    # Configuration
    config = {
        'data_root': 'dataset',
        'checkpoint_dir': 'checkpoints',
        'output_dir': 'outputs',
        'efficientnet_version': 'efficientnet-b0',
        #'hidden_dim': 128,
        'hidden_dim': 64,
        #'latent_dim': 32,
        'latent_dim': 16,
        #'batch_size': 8,
        'batch_size': 4,
        'learning_rate': 1e-4,
        #'learning_rate': 2e-4,
        'epochs': 200,
        'save_frequency': 10,
        'early_stopping_patience': 20,
        'preprocessing': True
    }

    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config['output_dir'], 'run.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    if args.mode in ['train', 'train_and_test']:
        logger.info('Starting training...')
        train(config)
        
    if args.mode in ['test', 'train_and_test']:
        logger.info('Starting testing...')
        metrics = test(config)
        
        logger.info('Final Results:')
        for metric, value in metrics.items():
            logger.info(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main()