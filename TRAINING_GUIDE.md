# YOLOv8 Training with Weights & Biases

This guide explains how to train a custom YOLOv8 model with experiment tracking using Weights & Biases (W&B).

## Prerequisites

1. Python 3.8 or higher
2. CUDA-capable GPU (recommended)
3. Weights & Biases account (https://wandb.ai)

## Setup

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Log in to Weights & Biases:
   ```bash
   wandb login
   ```
   You'll be prompted to enter your API key from https://wandb.ai/authorize

## Dataset Preparation

1. Organize your dataset in the following structure:
   ```
   datasets/
   └── custom_dataset/
       ├── images/
       │   ├── train/
       │   │   ├── image1.jpg
       │   │   └── ...
       │   └── val/
       │       ├── image101.jpg
       │       └── ...
       └── labels/
           ├── train/
           │   ├── image1.txt
           │   └── ...
           └── val/
               ├── image101.txt
               └── ...
   ```

2. Update the dataset configuration file at `data/custom_dataset.yaml`:
   - Set the correct path to your dataset
   - Update the class names and number of classes
   - Adjust other parameters as needed

## Training

Start training with default parameters:
```bash
python train.py --data data/custom_dataset.yaml --epochs 100 --imgsz 640 --batch 16
```

### Command Line Arguments

- `--data`: Path to dataset configuration file (required)
- `--epochs`: Number of training epochs (default: 50)
- `--imgsz`: Image size (default: 640)
- `--batch`: Batch size (default: 16)

## Monitoring Training

1. Open the W&B dashboard in your browser:
   ```bash
   wandb online
   ```
   Then visit https://wandb.ai/your-username/custom-object-detection

2. Track metrics in real-time:
   - Loss curves
   - mAP@0.5 and mAP@0.5:0.95
   - Precision/Recall
   - Learning rate schedule
   - Sample predictions

## Model Evaluation

After training, you'll find the best model at:
```
runs/detect/train/weights/best.pt
```

## Using the Trained Model

Use the trained model for inference with the provided `main.py` script:
```bash
python main.py --image path/to/image.jpg --model runs/detect/train/weights/best.pt
```

## Tips for Better Performance

1. **Data Quality**: Ensure high-quality, diverse training data
2. **Class Balance**: Balance the number of samples per class
3. **Augmentation**: Enable/disable augmentations based on your use case
4. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, etc.
5. **Early Stopping**: Monitor validation metrics to prevent overfitting

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size or image size
- **Slow Training**: Enable mixed precision with `--fp16`
- **W&B Connection Issues**: Check your internet connection and API key
