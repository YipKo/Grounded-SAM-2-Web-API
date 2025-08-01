import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Tuple, Optional
import warnings
import logging

sys.path.insert(0, os.path.join(os.getcwd(),"Grounded-SAM-2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert

warnings.filterwarnings("ignore")

class GroundedSAM2Segmenter:
    """Grounded-SAM-2 based segmenter"""
    
    def __init__(self, 
                 sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt",
                 sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 grounding_dino_config: str = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 grounding_dino_checkpoint: str = "gdino_checkpoints/groundingdino_swint_ogc.pth",
                 device: str = "cuda"):
        """
        Initialize Grounded-SAM-2 segmenter
        
        Args:
            sam2_checkpoint: SAM2 model checkpoint path
            sam2_model_config: SAM2 model config file path
            grounding_dino_config: Grounding DINO config file path
            grounding_dino_checkpoint: Grounding DINO checkpoint file path
            device: Device ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Get current working directory and switch to Grounded-SAM-2 directory
        self.working_dir = os.getcwd()
        self.grounded_sam2_dir = os.path.join(os.getcwd(), "Grounded-SAM-2")
        os.chdir(self.grounded_sam2_dir)
        
        try:
            # Initialize SAM2
            self.sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
            
            # Initialize Grounding DINO
            self.grounding_model = load_model(
                model_config_path=grounding_dino_config,
                model_checkpoint_path=grounding_dino_checkpoint,
                device=self.device
            )
        finally:
            # Restore original working directory
            os.chdir(self.working_dir)
        
        # Ensure all models use float32 precision
        if hasattr(self.grounding_model, 'float'):
            self.grounding_model = self.grounding_model.float()
        if hasattr(self.sam2_model, 'float'):
            self.sam2_model = self.sam2_model.float()
    
    def _preprocess_caption(self, caption: str) -> str:
        """Preprocess text prompt"""
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."
    
    def detect_objects(self, 
                      image: Union[str, np.ndarray, Image.Image],
                      text_prompts: List[str],
                      box_threshold: float = 0.35,
                      text_threshold: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Detect objects using Grounding DINO
        
        Args:
            image: Input image (path, numpy array or PIL image)
            text_prompts: List of text prompts
            box_threshold: Bounding box threshold
            text_threshold: Text threshold
            
        Returns:
            boxes: Detected bounding boxes (cxcywh format)
            logits: Confidence scores
            phrases: Detected phrases
        """
        # Load image
        if isinstance(image, str):
            # Convert to absolute path
            if not os.path.isabs(image):
                image = os.path.abspath(image)
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found at {image}")
            os.chdir(self.grounded_sam2_dir)
            try:
                image_source, image_tensor = load_image(image)
            finally:
                os.chdir(self.working_dir)
        elif isinstance(image, np.ndarray) or isinstance(image, Image.Image):
            # For numpy arrays or PIL images, save as temporary file first
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                if isinstance(image, np.ndarray):
                    # Assume input is RGB format
                    image_pil = Image.fromarray(image)
                else:
                    image_pil = image
                image_pil.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            try:
                os.chdir(self.grounded_sam2_dir)
                try:
                    image_source, image_tensor = load_image(tmp_path)
                finally:
                    os.chdir(self.working_dir)
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
        else:
            raise TypeError("image must be a file path (str), numpy array, or PIL Image")
        
        all_boxes, all_logits, all_phrases = [], [], []
        
        # Detect for each text prompt
        for text_prompt in text_prompts:
            try:
                os.chdir(self.grounded_sam2_dir)
                try:
                    # Disable autocast environment to ensure float32 usage
                    with torch.cuda.amp.autocast(enabled=False):
                        boxes, logits, phrases = predict(
                            model=self.grounding_model,
                            image=image_tensor,
                            caption=self._preprocess_caption(text_prompt),
                            box_threshold=box_threshold,
                            text_threshold=text_threshold,
                            device=self.device
                        )
                finally:
                    os.chdir(self.working_dir)

                if len(logits) == 0:
                    logging.warning(f"No objects detected for text: '{text_prompt}'")
                    continue
                
                # Select detection result with highest confidence
                best_idx = torch.argmax(logits)
                all_boxes.append(boxes[best_idx])
                all_logits.append(logits[best_idx])
                all_phrases.append(phrases[best_idx])
                
            except Exception as e:
                logging.error(f"Error detecting objects for '{text_prompt}': {e}")
                continue
        
        if len(all_boxes) == 0:
            return torch.empty(0, 4), torch.empty(0), []
        
        return torch.stack(all_boxes, dim=0), torch.stack(all_logits, dim=0), all_phrases
    
    def segment_objects(self,
                       image: Union[str, np.ndarray, Image.Image],
                       boxes: torch.Tensor) -> np.ndarray:
        """
        Segment objects using SAM2
        
        Args:
            image: Input image
            boxes: Bounding boxes (cxcywh format)
            
        Returns:
            masks: Segmentation mask arrays
        """
        # Process image
        if isinstance(image, str):
            image_source = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, np.ndarray):
            image_source = image
        elif isinstance(image, Image.Image):
            image_source = np.array(image.convert("RGB"))
        else:
            raise TypeError("image must be a file path (str), numpy array, or PIL Image")
        
        # Set image for SAM2 predictor
        self.sam2_predictor.set_image(image_source)
        
        # Convert bounding box format
        h, w, _ = image_source.shape
        boxes_scaled = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        # Set autocast according to official demo (only for SAM2 prediction)
        if self.device == "cuda":
            torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # Use SAM2 for prediction
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # Convert mask shape
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        return masks
    
    def get_masks_for_objects(self,
                             object_names: List[str],
                             image: Union[str, np.ndarray, Image.Image],
                             box_threshold: float = 0.35,
                             text_threshold: float = 0.25)  -> Union[Dict, None]:
        """
        Generate segmentation masks for object list
        
        Args:
            object_names: List of object names
            image: Input image
            box_threshold: Bounding box threshold
            text_threshold: Text threshold
            
        Returns:
            - np.ndarray: Single mask (0-255) if only one object (for backward compatibility)
            - dict: Complete results with masks, bboxes, scores, phrases if multiple objects
            - None: If no objects detected
        """
        try:
            # 1. Object detection
            boxes, logits, phrases = self.detect_objects(
                image=image,
                text_prompts=object_names,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            if len(boxes) == 0:
                print(f"Warning: Did not detect any of the objects: {object_names}")
                return None
            
            # 2. Segmentation
            masks = self.segment_objects(image=image, boxes=boxes)
            
            # Convert bounding box format (from cxcywh to xyxy) for output
            if isinstance(image, str):
                img_pil = Image.open(image)
            elif isinstance(image, np.ndarray):
                img_pil = Image.fromarray(image)
            else:
                img_pil = image
            img_array = np.array(img_pil)
            h, w = img_array.shape[:2]
            
            boxes_scaled = boxes * torch.Tensor([w, h, w, h])
            boxes_xyxy = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy")
            
            # 3. Organize results
            output_masks = {}
            output_bboxes = {}
            output_scores = {}
            output_phrases = {}
            
            for i, obj_name in enumerate(object_names):
                if i < len(masks):
                    # Convert mask to 0-255 format
                    mask_255 = (masks[i].astype(np.float32) * 255).astype(np.uint8)
                    output_masks[obj_name] = mask_255
                    
                    # Store detection information
                    output_bboxes[obj_name] = boxes_xyxy[i].cpu().numpy().tolist()
                    output_scores[obj_name] = logits[i].cpu().numpy().item()
                    output_phrases[obj_name] = phrases[i]
                else:
                    # Create empty mask
                    print(f"Warning: '{obj_name}' not found in segmentation results, creating empty mask.")
                    output_masks[obj_name] = np.zeros((h, w), dtype=np.uint8)
                    output_bboxes[obj_name] = [0, 0, 0, 0]
                    output_scores[obj_name] = 0.0
                    output_phrases[obj_name] = ""
            
            # 4. Return results - unified format for both single and multiple objects
            result = {}
            for obj_name in object_names:
                result[obj_name] = {
                    "success": obj_name in output_masks and output_masks[obj_name].max() > 0,
                    "mask": output_masks[obj_name],
                    "bbox": output_bboxes[obj_name],
                    "score": output_scores[obj_name],
                    "phrase": output_phrases[obj_name]
                }
            return result
                
        except Exception as e:
            print(f"Error in get_masks_for_objects: {e}")
            return None

if __name__ == "__main__":
    print("Starting deer segmentation test...")
    
    # Check if deer.png file exists
    deer_image_path = "assets/deer.png"
    if not os.path.exists(deer_image_path):
        print(f"Error: {deer_image_path} file not found")
        raise FileNotFoundError(f"{deer_image_path} file not found")
    
    try:
        # Initialize segmenter
        print("Initializing GroundedSAM2Segmenter...")
        segmenter = GroundedSAM2Segmenter()
        
        # Segment deer
        print("Segmenting deer...")
        result = segmenter.get_masks_for_objects(
            object_names=["deer"],
            image=deer_image_path,
            box_threshold=0.35,
            text_threshold=0.25
        )
        
        if result is not None:
            # New unified format: always returns dict
            if "deer" in result:
                obj_info = result["deer"]
                if obj_info.get("success"):
                    mask = obj_info["mask"]
                    print(f"Segmentation successful! Mask shape: {mask.shape}")
                    print(f"Mask data type: {mask.dtype}")
                    print(f"Mask value range: {mask.min()} - {mask.max()}")
                    
                    # Save segmentation result
                    output_path = "deer_mask.png"
                    mask_image = Image.fromarray(mask)
                    mask_image.save(output_path)
                    print(f"Segmentation mask saved to: {output_path}")
                    
                    # Create visualization image (overlay original image with mask)
                    original_image = Image.open(deer_image_path).convert("RGB")
                    original_array = np.array(original_image)
                    
                    # Create colored mask
                    colored_mask = np.zeros_like(original_array)
                    colored_mask[mask > 0] = [255, 0, 0]  # Red mask
                    
                    # Overlay images
                    alpha = 0.5
                    overlay = (original_array * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
                    overlay_image = Image.fromarray(overlay)
                    overlay_path = "deer_overlay.png"
                    overlay_image.save(overlay_path)
                    print(f"Overlay image saved to: {overlay_path}")
                else:
                    print("Segmentation failed: deer not detected")
            else:
                print("Segmentation failed: deer not found in results")
            
        else:
            print("Segmentation failed: deer not detected or error occurred during segmentation")
            
    except Exception as e:
        print(f"Error occurred during testing: {e}")
        import traceback
        traceback.print_exc()