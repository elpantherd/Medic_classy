# src/preprocessing.py
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

def trim_borders(pil_img, max_border_ratio=0.15):
    # Remove large, uniform borders quickly
    img = np.array(pil_img.convert('L'))
    H, W = img.shape
    # Find rows/cols with content
    row_nonzero = np.where(img < 250)[0]
    col_nonzero = np.where(img < 250)[0]
    if row_nonzero.size == 0 or col_nonzero.size == 0:
        return pil_img
    # Simple crop to non-white area
    ys = np.where(img.min(axis=1) < 250)[0]
    xs = np.where(img.min(axis=0) < 250)[0]
    if ys.size==0 or xs.size==0:
        return pil_img
    y0, y1 = ys[0], ys[-1]
    x0, x1 = xs[0], xs[-1]
    # Avoid over-cropping
    if (y1-y0)/H < (1 - max_border_ratio) and (x1-x0)/W < (1 - max_border_ratio):
        return pil_img
    return pil_img.crop((x0, y0, x1, y1))

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img = trim_borders(img)
        if img.mode != 'L':
            img = img.convert('L')
        img_np = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        clahe_img = clahe.apply(img_np)
        return Image.fromarray(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB))

def get_transform():
    return transforms.Compose([
        CLAHETransform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
