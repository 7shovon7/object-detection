# Object Detection with Ultralytics YOLOv8

This is a simple and efficient object detection program using Ultralytics YOLOv8, which automatically handles model downloading and provides state-of-the-art object detection capabilities.

## Features

- Supports multiple YOLOv8 model sizes (nano, small, medium, large, xlarge)
- Automatic model downloading and caching
- GPU acceleration support (if CUDA is available)
- Clean and easy-to-use Python API
- Automatic output organization with timestamps
- Creates necessary directories automatically

## Prerequisites

- Python 3.8 or higher
- PyTorch (automatically installed with ultralytics)
- OpenCV

## Installation

1. Clone this repository or download the files.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To detect objects in an image, run:

```bash
python object_detection.py --image path/to/your/image.jpg --output output.jpg
```

### Arguments

- `--image`: Path to the input image (required)
- `--model`: YOLOv8 model to use (default: 'yolov8n.pt')
  - Options: 'yolov8n.pt' (nano), 'yolov8s.pt' (small), 'yolov8m.pt' (medium), 'yolov8l.pt' (large), 'yolov8x.pt' (xlarge)
- `--confidence`: Minimum confidence threshold (0-1, default: 0.5)
- `--output`: (Optional) Custom path to save the output image. If not provided, saves to: `images/output/inputname_detection_TIMESTAMP.jpg`

### Examples

1. Basic usage with default model (YOLOv8n):

   ```bash
   python object_detection.py --image sample.jpg
   ```

2. Using a larger model with custom confidence threshold:

   ```bash
   python object_detection.py --image sample.jpg --model yolov8m.pt --confidence 0.7 --output detected_objects.jpg
   ```

## How It Works

1. The program automatically downloads the specified YOLOv8 model if not already cached (saved in `~/.cache/ultralytics/`).
2. It processes the input image through the neural network.
3. It filters detections based on the confidence threshold.
4. It draws bounding boxes around detected objects with their class labels and confidence scores.
5. It saves the output image with detections in the `images/output/` directory with a timestamped filename (if no custom output path is provided).

## Directory Structure

```text
project/
├── object_detection.py  # Main detection script
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── images/
│   ├── input/          # Directory for input images (optional)
│   └── output/         # Automatically created for output images
└── docs/
    └── JOURNEY.md     # Development journey log
```

## Output Files

Output files are automatically named using this format:

```text
images/output/inputname_detection_YYYYMMDD_HHMMSS.jpg
```

For example, processing `my_photo.jpg` would create something like:

```text
images/output/my_photo_detection_20250608_214322.jpg
```

## Performance Notes

- The first run will download the model (about 6MB for the nano version, up to 130MB for the xlarge version).
- For better performance, the program will automatically use GPU if CUDA is available.
- Smaller models (like 'yolov8n.pt') are faster but less accurate than larger ones.

## License

This project is open source and available under the MIT License.
