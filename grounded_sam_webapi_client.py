#!/usr/bin/env python3
"""
Grounded-SAM-2 Web API Client Example
Demonstrates how to use the Web API for image segmentation
"""

import os
import sys
import base64
import io
import json
import requests
import numpy as np
import cv2
from PIL import Image

class GroundedSAMClient:
    """Grounded-SAM-2 Web API Client"""
    
    def __init__(self, base_url="http://localhost:5000", token=None):
        """
        Initialize client
        
        Args:
            base_url: API server address
            token: API token for authentication
        """
        self.base_url = base_url
        self.token = token or os.environ.get('API_TOKEN', 'your-secure-token-here')
        self.headers = {'Authorization': f'Bearer {self.token}'}
        
        # Test connection and token validation
        health = self.health_check()
        if not health.get('authenticated'):
            raise ConnectionError("Invalid token or unable to authenticate with server")
        print(f"✅ Connected to server with valid token")
    
    def health_check(self):
        """Health check with token validation"""
        try:
            response = requests.get(f"{self.base_url}/health", headers=self.headers, timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def image_to_base64(self, image_path_or_array):
        """Convert image to base64 string"""
        if isinstance(image_path_or_array, str):
            # Read from file path
            with open(image_path_or_array, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        elif isinstance(image_path_or_array, np.ndarray):
            # Convert from numpy array
            if image_path_or_array.dtype != np.uint8:
                image_path_or_array = image_path_or_array.astype(np.uint8)
            
            pil_img = Image.fromarray(image_path_or_array)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            raise TypeError("image_path_or_array must be a file path or numpy array")
    
    def base64_to_image(self, base64_str):
        """Convert base64 string to numpy array"""
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        return np.array(img)
    
    def segment_object(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """
        Single object segmentation
        
        Args:
            image: Image path or numpy array
            text_prompt: Object description text
            box_threshold: Bounding box threshold
            text_threshold: Text threshold
            
        Returns:
            dict: Segmentation result
        """
        try:
            # Convert image to base64
            image_base64 = self.image_to_base64(image)
            
            # Construct request data
            data = {
                "image": image_base64,
                "text_prompt": text_prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold
            }
            
            # Send request
            response = requests.post(
                f"{self.base_url}/segment",
                json=data,
                headers=self.headers,
                timeout=60
            )
            
            result = response.json()
            
            # If successful, decode masks
            if result.get('success') and result.get('masks'):
                masks = []
                for mask_base64 in result['masks']:
                    mask = self.base64_to_image(mask_base64)
                    masks.append(mask)
                result['masks_decoded'] = masks
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def batch_segment_objects(self, image, text_prompts, box_threshold=0.3, text_threshold=0.25):
        """
        Batch segment multiple objects in the image
        
        Args:
            image: Input image (numpy array or file path)
            text_prompts: List of text prompts describing objects to segment
            box_threshold: Box confidence threshold
            text_threshold: Text confidence threshold
            
        Returns:
            dict: API response with decoded masks, bboxes, scores, and phrases
        """
        try:
            # Convert image to base64
            image_base64 = self.image_to_base64(image)
            
            # Construct request data
            data = {
                "image": image_base64,
                "text_prompts": text_prompts,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold
            }
            
            # Send request
            response = requests.post(
                f"{self.base_url}/batch_segment",
                json=data,
                headers=self.headers,
                timeout=60
            )
            
            result = response.json()
            
            # If successful, decode masks
            if result.get('success'):
                masks_decoded = {}
                
                # Handle new format where each object has its own structure
                for obj_name in text_prompts:
                    if obj_name in result:
                        obj_data = result[obj_name]
                        if isinstance(obj_data, dict) and obj_data.get('success'):
                            masks = obj_data.get('masks', [])
                            if masks:
                                mask_b64 = masks[0]  # Take first mask
                                mask_data = base64.b64decode(mask_b64)
                                mask_array = np.frombuffer(mask_data, dtype=np.uint8)
                                mask_decoded = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
                                masks_decoded[obj_name] = mask_decoded
                
                result['masks_decoded'] = masks_decoded
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Grounded-SAM-2 Web API Client')
    parser.add_argument('--url', default='http://localhost:5000',
                        help='API server address')
    parser.add_argument('--token', help='API token (overrides API_TOKEN env var)')
    
    args = parser.parse_args()
    
    # 1. Initialize client
    client = GroundedSAMClient(args.url, token=args.token)
    
    print("🧪 Grounded-SAM-2 Web API Client Test")
    
    # 2. Read test image
    # Or use real image
    test_image = "deer.png"
    test_image = cv2.imread(test_image)
    
    # 3. Single object segmentation test
    print("\n3️⃣ Testing single object segmentation...")
    result = client.segment_object(
        image=test_image,
        text_prompt="deer",
        box_threshold=0.3,
        text_threshold=0.25
    )
    
    print(f"Segmentation result: {result.get('success')}")
    if result.get('success'):
        print(f"  - Detected {len(result.get('bboxes', []))} bounding boxes")
        print(f"  - Generated {len(result.get('masks', []))} masks")
        print(f"  - Detection phrases: {result.get('phrases', [])}")
        print(f"  - Confidence scores: {result.get('scores', [])}")
    else:
        print(f"  - Error: {result.get('error')}")
    
    # 4. Batch segmentation test
    print("\n4️⃣ Testing batch object segmentation...")
    batch_result = client.batch_segment_objects(
        image=test_image,
        text_prompts=["deer", "house"],
        box_threshold=0.3,
        text_threshold=0.25
    )
    
    print(f"Batch segmentation result: {batch_result.get('success')}")
    if batch_result.get('success'):
        detected_objects = []
        for obj_name in ["deer", "house"]:
            if obj_name in batch_result:
                obj_data = batch_result[obj_name]
                if isinstance(obj_data, dict) and obj_data.get('success'):
                    detected_objects.append(obj_name)
                    bboxes = obj_data.get('bboxes', [])
                    scores = obj_data.get('scores', [])
                    phrases = obj_data.get('phrases', [])
                    print(f"  - {obj_name}: {len(bboxes)} detections")
                    if scores:
                        print(f"    - Scores: {scores}")
                    if phrases:
                        print(f"    - Phrases: {phrases}")
        
        print(f"  - Detected objects: {detected_objects}")
    else:
        print(f"  - Error: {batch_result.get('error')}")
    
    print("\n✅ Client test completed!")

