import torch
import torch.nn as nn
import open_clip

class CLIPClassifier(nn.Module):
    """
    Zero-shot classifier using CLIP with better prompts for medical images.
    """
    def __init__(self, device='cpu'):
        super(CLIPClassifier, self).__init__()
        self.device = device
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.model.to(self.device)
            self.model.eval()
            print("CLIP model loaded successfully.")
        except Exception as e:
            print(f"Failed to load CLIP model: {e}")
            raise

    def forward(self, pil_image, text_prompts):
        """
        Calculate similarity between image and text prompts.
        """
        # Preprocess image
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer(text_prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize and compute similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
        return similarity.squeeze(0)
