import torch
from PIL import Image
from tqdm import tqdm
import logging
import time
from .model import CLIPClassifier

logger = logging.getLogger(__name__)

class ImageClassifier:
    """
    Handles loading the CLIP model and performing zero-shot classification.
    """
    def __init__(self, device=None, batch_size=32, is_url=False):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        self.batch_size = batch_size
        
        try:
            self.model = CLIPClassifier(device=self.device)
            # Conditionally set prompts based on input type
            if is_url:
                # Your custom prompt for websites (replace with your exact desired prompt)
                self.prompts = [
                    "a diagnostic medical image from a website, such as an online pathology slide, tissue micrograph, or web-hosted scan",
                    "a non-medical image from a website, such as a logo, icon, or everyday photo"
                ]
            else:
                # Existing prompt for PDFs (unchanged from your working version)
                self.prompts = [
                    "a diagnostic medical image, such as a CT scan, MRI, x-ray, or pathology slide",
                    "a photograph of an everyday object, person, landscape, or non-medical diagram"
                ]
            self.class_names = ["medical", "non-medical"]
            logger.info(f"CLIP ImageClassifier initialized on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize CLIP classifier: {e}", exc_info=True)
            raise

    def classify_images(self, images, source_id, metadata_list=None):
        """
        Classifies a list of PIL images using CLIP zero-shot prediction.
        """
        results = {
            "source": source_id,
            "processing_summary": {},
            "classifications": []
        }
        
        total_images = len(images)
        if total_images == 0:
            return results

        start_time = time.time()
        
        for i, pil_image in enumerate(tqdm(images, desc="Classifying with CLIP")):
            try:
                # Get probabilities for the prompts
                probs = self.model(pil_image, self.prompts)
                
                # Get the highest probability and its corresponding index
                confidence, pred_index = torch.max(probs, 0)
                
                classification_result = {
                    "image_index": i,
                    "classification": self.class_names[pred_index],
                    "confidence": round(confidence.item(), 4)
                }
                
                if metadata_list and i < len(metadata_list):
                    classification_result.update(metadata_list[i])
                    
                results["classifications"].append(classification_result)
                
            except Exception as e:
                logger.warning(f"Failed to classify image {i}: {e}")
                continue
            
        end_time = time.time()
        processing_time = end_time - start_time
        images_per_second = total_images / processing_time if processing_time > 0 else float('inf')
        
        results["processing_summary"] = {
            "total_images_processed": len(results["classifications"]),
            "processing_time_seconds": round(processing_time, 4),
            "images_per_second": round(images_per_second, 2),
            "model_used": "CLIP Zero-Shot Classifier (ViT-B/32)",
            "device": self.device
        }

        return results
