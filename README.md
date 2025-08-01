# Grounded-SAM-2 Web API

A web API service based on Grounded-SAM-2 for semantic segmentation and object detection. This project provides REST API interfaces for image segmentation using natural language prompts.

## Features

- 🚀 **Web API Interface**: Easy-to-use REST API for image segmentation
- 🎯 **Natural Language Prompts**: Segment objects using text descriptions
- 🔄 **Batch Processing**: Support for multiple object segmentation in one request
- 📦 **Multiple Input Formats**: Support for file paths, numpy arrays, and PIL images
- 🌐 **Cross-Platform**: Works on Linux, Windows, and macOS
- 📋 **Client SDK**: Python client library included

## Installation

### Prerequisites

- Python 3.10
- CUDA 12.1+ (recommended for GPU acceleration)

### Step 1: Clone Grounded-SAM-2

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
```

### Step 2: Install Dependencies

```bash
# Set CUDA environment (adjust path according to your CUDA installation)
export CUDA_HOME=/path/to/cuda-12.1/

cd Grounded-SAM-2/

# Install PyTorch with CUDA support
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install Grounded-SAM-2
pip install -e .
pip install --no-build-isolation -e grounding_dino

# Install additional dependencies for web API
pip install flask iopath opencv-contrib-python-headless
pip install -r grounding_dino/requirements.txt
```

### Step 3: Download Model Checkpoints

```bash
# Download SAM2 checkpoints
cd checkpoints
bash download_ckpts.sh
cd ..

# Download Grounding DINO checkpoints
cd gdino_checkpoints
bash download_ckpts.sh
cd ..
```

### Step 4: Clone This Repository

```bash
cd ..
git clone https://github.com/yourusername/Grounded-SAM_web-api.git
cd Grounded-SAM_web-api
```

## Quick Start

### 1. Test Grounded-SAM-2 Installation

First, ensure Grounded-SAM-2 is working correctly by running the test script:

```bash
python segmentation.py
```

This will:
- Load the Grounded-SAM-2 models
- Segment a deer from `deer.png`
- Save the segmentation mask as `deer_mask.png`
- Save an overlay visualization as `deer_overlay.png`

### 2. Start the Web API Server

Once the test passes, start the web API server:

```bash
python grounded_sam_web_api.py
```

The server will start on `http://localhost:5000` by default.

### 3. Test the API

Run the client example to test the API:

```bash
python grounded_sam_client.py
```

## API Usage

### Health Check

```bash
curl http://localhost:5000/health
```

### Single Object Segmentation

```bash
curl -X POST http://localhost:5000/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image",
    "text_prompt": "deer",
    "box_threshold": 0.3,
    "text_threshold": 0.25
  }'
```

### Batch Object Segmentation

```bash
curl -X POST http://localhost:5000/batch_segment \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image",
    "text_prompts": ["deer", "tree", "grass"],
    "box_threshold": 0.3,
    "text_threshold": 0.25
  }'
```

## Python Client Usage

```python
from grounded_sam_client import GroundedSAMClient

# Initialize client
client = GroundedSAMClient("http://localhost:5000")

# Single object segmentation
result = client.segment_object(
    image="path/to/image.jpg", # or Image in numpy array format
    text_prompt="deer",
    box_threshold=0.3,
    text_threshold=0.25
)

# Batch segmentation
batch_result = client.batch_segment_objects(
    image="path/to/image.jpg", # or Image in numpy array format
    text_prompts=["deer", "tree", "grass"]
)
```

## API Reference

### POST /segment

Segment a single object in an image.

**Request Body:**
```json
{
  "image": "base64_encoded_image_string",
  "text_prompt": "object_description",
  "box_threshold": 0.3,
  "text_threshold": 0.25
}
```

**Response:**
```json
{
  "success": true,
  "bboxes": [[x1, y1, x2, y2], ...],
  "masks": ["base64_encoded_mask", ...],
  "phrases": ["detected_phrase", ...],
  "scores": [confidence_score, ...]
}
```

### POST /batch_segment

Segment multiple objects in an image.

**Request Body:**
```json
{
  "image": "base64_encoded_image_string",
  "text_prompts": ["object1", "object2", "object3"],
  "box_threshold": 0.3,
  "text_threshold": 0.25
}
```

**Response:**
```json
{
  "success": true,
  "object1": {
    "success": true,
    "bboxes": [[x1, y1, x2, y2], ...],
    "masks": ["base64_encoded_mask", ...],
    "phrases": ["detected_phrase", ...],
    "scores": [confidence_score, ...]
  },
  "object2": {
    "success": true,
    "bboxes": [[x1, y1, x2, y2], ...],
    "masks": ["base64_encoded_mask", ...],
    "phrases": ["detected_phrase", ...],
    "scores": [confidence_score, ...]
  },
  "object3": {
    "success": true,
    "bboxes": [[x1, y1, x2, y2], ...],
    "masks": ["base64_encoded_mask", ...],
    "phrases": ["detected_phrase", ...],
    "scores": [confidence_score, ...]
  }
}

```

## Configuration

You can customize the server configuration by modifying the parameters in `grounded_sam_web_api.py` or by passing command-line arguments:

```bash
python grounded_sam_web_api.py --host 0.0.0.0 --port 8080 --debug
```

## File Structure

```
Grounded-SAM_web-api/
├── segmentation.py          # Core segmentation logic
├── grounded_sam_web_api.py  # Flask web server
├── grounded_sam_client.py   # Python client library
├── deer.png                 # Test image
├── README.md               # This file
└── Grounded-SAM-2/         # Grounded-SAM-2 repository (git submodule)
```



## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the original Grounded-SAM-2 repository for its licensing terms.

## Acknowledgments

- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) - The underlying segmentation model
- [SAM2](https://github.com/facebookresearch/sam2) - Segment Anything Model 2
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) - Object detection with natural language

## Citation

If you use this project in your research, please cite the original papers:

```bibtex

```


