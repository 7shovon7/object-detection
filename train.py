import os
import wandb
import yaml
from pathlib import Path
from ultralytics import YOLO

def setup_wandb():
    """Initialize Weights & Biases for experiment tracking."""
    # Initialize W&B run
    run = wandb.init(
        project="custom-object-detection",
        name=f"yolov8x-{wandb.util.generate_id()}",
        config={
            "framework": "YOLOv8",
            "task": "object-detection",
        }
    )
    return run

def train_model(data_yaml, epochs=50, imgsz=640, batch=16):
    """
    Train YOLOv8 model with W&B integration.
    
    Args:
        data_yaml (str): Path to dataset YAML file
        epochs (int): Number of training epochs
        imgsz (int): Image size
        batch (int): Batch size
    """
    # Setup W&B
    run = setup_wandb()
    
    try:
        # Load a pretrained YOLOv8x model
        model = YOLO('yolov8x.pt')
        
        # Train the model with W&B integration
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project='object-detection',
            name=f'run-{wandb.run.id}',
            save_period=10,  # Save checkpoint every 10 epochs
            device=0,  # Use GPU 0
            workers=8,
            optimizer='auto',
            lr0=0.01,  # Initial learning rate
            lrf=0.01,  # Final learning rate (lr0 * lrf)
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,  # box loss gain
            cls=0.5,  # cls loss gain
            dfl=1.5,  # dfl loss gain
            hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
            hsv_s=0.7,  # image HSV-Saturation augmentation (fraction)
            hsv_v=0.4,  # image HSV-Value augmentation (fraction)
            degrees=0.0,  # image rotation (+/- deg)
            translate=0.1,  # image translation (+/- fraction)
            scale=0.5,  # image scale (+/- gain)
            shear=0.0,  # image shear (+/- deg)
            perspective=0.0,  # image perspective (+/- fraction), range 0-0.001
            flipud=0.0,  # image flip up-down (probability)
            fliplr=0.5,  # image flip left-right (probability)
            mosaic=1.0,  # image mosaic (probability)
            mixup=0.0,  # image mixup (probability)
            copy_paste=0.0,  # segment copy-paste (probability)
        )
        
        # Log final metrics
        if results:
            wandb.log({
                'metrics/mAP50-95': results.results_dict['metrics/mAP50-95(B)'],
                'metrics/mAP50': results.results_dict['metrics/mAP50(B)'],
                'metrics/precision': results.results_dict['metrics/precision(B)'],
                'metrics/recall': results.results_dict['metrics/recall(B)'],
                'lr/pg0': results.results_dict['lr/pg0'],
                'lr/pg1': results.results_dict['lr/pg1'],
                'lr/pg2': results.results_dict['lr/pg2'],
            })
            
    except Exception as e:
        print(f"Error during training: {e}")
        wandb.alert("Training Failed", str(e))
        raise
    finally:
        # Ensure W&B run is finished
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 model with W&B integration')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    # Login to W&B (will prompt for API key if not already logged in)
    wandb.login()
    
    # Start training
    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
