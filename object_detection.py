import argparse
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np
from datetime import datetime
import os

def ensure_output_dir():
    """Ensure the output directory exists, create if it doesn't."""
    output_dir = Path("images/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def generate_output_filename(input_path):
    """Generate a timestamped output filename based on the input filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_stem = Path(input_path).stem
    output_dir = ensure_output_dir()
    return output_dir / f"{input_stem}_detection_{timestamp}.jpg"

class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """
        Initialize the object detector with Ultralytics YOLO.
        The model will be automatically downloaded if not found locally.
        
        Args:
            model_name (str): Name of the YOLO model to use (e.g., 'yolov8n.pt', 'yolov8s.pt', etc.)
        """
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load YOLO model (will download if not found)
        self.model = YOLO(model_name).to(self.device)
        
        # Get class names
        self.classes = self.model.names
        
        # Generate random colors for each class
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
    
    def detect_objects(self, image_path, confidence_threshold=0.5):
        """
        Detect objects in an image.
        
        Args:
            image_path (str): Path to the input image
            confidence_threshold (float): Minimum confidence threshold for detections
            
        Returns:
            tuple: (image with detections, list of detections)
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        # Run inference
        results = self.model(image, conf=confidence_threshold, device=self.device)
        
        # Process results
        detections = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                
                # Get class name and color
                class_name = self.classes[cls_id]
                color = self.colors[cls_id % len(self.colors)]
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label and confidence
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'box': (x1, y1, w, h)
                })
        
        return image, detections


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Object Detection using Ultralytics YOLO")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model to use (e.g., 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum probability to filter weak detections (0-1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output image. If not provided, will save to images/output/ with timestamp"
    )

    args = parser.parse_args()

    try:
        print(f"Initializing YOLO model: {args.model}")
        print("This may take a moment as the model is downloaded if not already cached...")
        
        # Initialize detector
        detector = ObjectDetector(args.model)

        print("Detecting objects...")
        # Detect objects
        output_image, detections = detector.detect_objects(args.image, args.confidence)

        # Determine output path
        if args.output is None:
            output_path = generate_output_filename(args.image)
        else:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save output image
        cv2.imwrite(str(output_path), output_image)
        print(f"\nDetection completed! Results saved to {output_path}")

        # Print detections
        if detections:
            print("\nDetected objects:")
            for i, detection in enumerate(detections, 1):
                print(f"{i}. {detection['class']} - Confidence: {detection['confidence']:.2f}")
        else:
            print("\nNo objects detected with the given confidence threshold.")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
