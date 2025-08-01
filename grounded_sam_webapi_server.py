#!/usr/bin/env python3
"""
Grounded-SAM-2 Web API Server with token authentication
"""

import base64
import io
import json
import os
import traceback
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from functools import wraps
import torch

# Import segmentation module
from segmentation import GroundedSAM2Segmenter

app = Flask(__name__)

# Global segmenter instance
segmenter = None

# API Token (set via environment variable or use default)
API_TOKEN = os.environ.get('API_TOKEN', 'your-secure-token-here')

def require_token(f):
    """Token validation decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'success': False, 'error': 'Missing Authorization header'}), 401
        
        # Support both "Bearer token" and "token" formats
        if token.startswith('Bearer '):
            token = token[7:]
        
        if token != API_TOKEN:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def init_segmenter():
    """Initialize segmenter"""
    global segmenter
    if segmenter is None:
        try:
            segmenter = GroundedSAM2Segmenter()
            print("✅ Grounded-SAM-2 segmenter initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize segmenter: {e}")
            raise e

def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy image array"""
    try:
        # Remove possible data URL prefix
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB format numpy array
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Unable to parse base64 image: {e}")

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image array to base64 string"""
    try:
        # Ensure image is uint8 format
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Convert to PIL image
        if len(image.shape) == 3:
            pil_img = Image.fromarray(image, 'RGB')
        else:
            pil_img = Image.fromarray(image, 'L')
        
        # Encode as PNG format base64
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64
    except Exception as e:
        raise ValueError(f"Unable to convert image to base64: {e}")

def boxes_to_list(boxes: torch.Tensor) -> List[List[float]]:
    """Convert torch tensor bounding boxes to list format"""
    if boxes.numel() == 0:
        return []
    return boxes.cpu().numpy().tolist()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with optional token validation"""
    # Check if token is provided for authenticated health check
    token = request.headers.get('Authorization')
    if token:
        # Support both "Bearer token" and "token" formats
        if token.startswith('Bearer '):
            token = token[7:]
        
        if token != API_TOKEN:
            return jsonify({
                'status': 'healthy',
                'service': 'Grounded-SAM-2 Web API',
                'authenticated': False,
                'error': 'Invalid token'
            }), 401
        
        return jsonify({
            'status': 'healthy',
            'service': 'Grounded-SAM-2 Web API',
            'authenticated': True,
            'segmenter_loaded': segmenter is not None
        })
    else:
        return jsonify({
            'status': 'healthy',
            'service': 'Grounded-SAM-2 Web API',
            'authenticated': False
        })

@app.route('/segment', methods=['POST'])
@require_token
def segment_objects():
    """
    Object segmentation endpoint
    
    Request format:
    {
        "image": "base64_encoded_image",
        "text_prompt": "object_description_to_detect",
        "box_threshold": 0.3,
        "text_threshold": 0.25
    }
    
    Response format:
    {
        "success": true,
        "bboxes": [[x1, y1, x2, y2], ...],  # Bounding boxes in xyxy format
        "masks": ["base64_encoded_mask", ...],
        "phrases": ["detected_phrase", ...],
        "scores": [confidence_score, ...]
    }
    """
    try:
        # Check if segmenter is initialized
        if segmenter is None:
            return jsonify({
                'success': False,
                'error': 'Segmenter not initialized'
            }), 500
        
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Empty request data'
            }), 400
        
        # Get parameters
        image_base64 = data.get('image')
        text_prompt = data.get('text_prompt', 'chair.')
        box_threshold = data.get('box_threshold', 0.3)
        text_threshold = data.get('text_threshold', 0.25)
        
        if not image_base64:
            return jsonify({
                'success': False,
                'error': 'Missing image data'
            }), 400
        
        # Convert base64 image to numpy array
        try:
            image_array = base64_to_image(image_base64)
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
        
        # Perform object detection
        boxes, logits, phrases = segmenter.detect_objects(
            image=image_array,
            text_prompts=[text_prompt],
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        if len(boxes) == 0:
            return jsonify({
                'success': True,
                'bboxes': [],
                'masks': [],
                'phrases': [],
                'scores': [],
                'message': f'No objects detected: {text_prompt}'
            })
        
        # Perform segmentation
        masks = segmenter.segment_objects(image=image_array, boxes=boxes)
        
        # Convert bounding box format (from cxcywh to xyxy)
        h, w, _ = image_array.shape
        boxes_scaled = boxes * torch.Tensor([w, h, w, h])
        from torchvision.ops import box_convert
        boxes_xyxy = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy")
        
        # Convert masks to base64
        masks_base64 = []
        for mask in masks:
            # Convert boolean mask to 0-255 uint8 image
            mask_uint8 = (mask.astype(np.float32) * 255).astype(np.uint8)
            mask_base64 = image_to_base64(mask_uint8)
            masks_base64.append(mask_base64)
        
        return jsonify({
            'success': True,
            'bboxes': boxes_to_list(boxes_xyxy),
            'masks': masks_base64,
            'phrases': phrases,
            'scores': logits.cpu().numpy().tolist() if len(logits) > 0 else []
        })
        
    except Exception as e:
        print(f"Segmentation processing error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/batch_segment', methods=['POST'])
@require_token
def batch_segment_objects():
    """
    Batch object segmentation endpoint
    
    Request format:
    {
        "image": "base64_encoded_image",
        "text_prompts": ["chair", "table", "person"],
        "box_threshold": 0.3,
        "text_threshold": 0.25
    }
    
    Response format:
    {
        "success": true,
        "deer": {
            "success": true,
            "bboxes": [[x1, y1, x2, y2], ...],
            "masks": ["base64_encoded_mask", ...],
            "phrases": ["detected_phrase", ...],
            "scores": [confidence_score, ...]
        },
        "tree": {
            "success": true,
            "bboxes": [[x1, y1, x2, y2], ...],
            "masks": ["base64_encoded_mask", ...],
            "phrases": ["detected_phrase", ...],
            "scores": [confidence_score, ...]
        }
    }
    """
    try:
        if segmenter is None:
            return jsonify({
                'success': False,
                'error': 'Segmenter not initialized'
            }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Empty request data'
            }), 400
        
        image_base64 = data.get('image')
        text_prompts = data.get('text_prompts', [])
        box_threshold = data.get('box_threshold', 0.3)
        text_threshold = data.get('text_threshold', 0.25)
        
        if not image_base64:
            return jsonify({
                'success': False,
                'error': 'Missing image data'
            }), 400
        
        if not text_prompts:
            return jsonify({
                'success': False,
                'error': 'Missing text prompts'
            }), 400
        
        # Convert image
        image_array = base64_to_image(image_base64)
        
        # Use existing API to get masks
        result = segmenter.get_masks_for_objects(
            object_names=text_prompts,
            image=image_array,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        if result is None:
            return jsonify({
                'success': True,
                'message': 'No objects detected'
            })
        
        # Process results - simplified unified format handling
        response_data: Dict[str, Any] = {'success': True}
        
        # Handle unified format from segmentation.py (always dict now)
        for obj_name in text_prompts:
            if obj_name in result and isinstance(result[obj_name], dict):
                obj_info = result[obj_name]
                success = bool(obj_info.get("success", False))
                if success and "mask" in obj_info:
                    response_data[obj_name] = {
                        "success": True,
                        "masks": [image_to_base64(obj_info["mask"])],
                        "bboxes": [obj_info.get("bbox", [0, 0, 0, 0])],
                        "scores": [float(obj_info.get("score", 0.0))],
                        "phrases": [str(obj_info.get("phrase", ""))]
                    }
                else:
                    response_data[obj_name] = {
                        "success": False,
                        "masks": [],
                        "bboxes": [],
                        "scores": [],
                        "phrases": []
                    }
            else:
                response_data[obj_name] = {
                    "success": False,
                    "masks": [],
                    "bboxes": [],
                    "scores": [],
                    "phrases": []
                }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Batch segmentation processing error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Start web server"""
    print("🚀 Starting Grounded-SAM-2 Web API server...")
    print(f"🔑 API Token: {API_TOKEN}")
    
    # Initialize segmenter
    init_segmenter()
    
    print(f"🌐 Server address: http://{host}:{port}")
    print("📋 Available endpoints:")
    print("  - GET  /health - Health check (token optional)")
    print("  - POST /segment - Single object segmentation (token required)")
    print("  - POST /batch_segment - Batch object segmentation (token required)")
    print("💡 Set API_TOKEN environment variable to change the default token")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Grounded-SAM-2 Web API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Server host address')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--token', help='API token (overrides API_TOKEN env var)')
    
    args = parser.parse_args()
    
    # Override token if provided via command line
    if args.token:
        API_TOKEN = args.token
    
    run_server(host=args.host, port=args.port, debug=args.debug)
