# src/pdf_cropper.py
import io
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import cv2

def render_page(pdf_path, page_index, dpi=200):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def find_visual_blocks(pil_img, min_block_area_ratio=0.02, blur=3, thresh=180):
    # Convert to grayscale, binarize, find contours to detect “image-like” blocks
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    # Invert to make dark content white regions
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    # Morph to merge nearby components
    kernel = np.ones((5,5), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = bw.shape
    min_area = H*W*min_block_area_ratio
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h >= min_area and w>40 and h>40:
            # Expand slightly to avoid tight crop cutting
            pad = int(0.02*max(W,H))
            x0 = max(0, x-pad); y0 = max(0, y-pad)
            x1 = min(W, x+w+pad); y1 = min(H, y+h+pad)
            boxes.append((x0,y0,x1,y1))
    # Sort boxes top-to-bottom
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

def crop_blocks(pil_img, boxes):
    crops = []
    for (x0,y0,x1,y1) in boxes:
        crops.append(pil_img.crop((x0,y0,x1,y1)))
    return crops

def extract_crops_from_pdf(pdf_path, dpi=200):
    # Return list of (page_number, index_on_page, crop_image)
    all_crops = []
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()
    for p in range(n_pages):
        page_img = render_page(pdf_path, p, dpi=dpi)
        boxes = find_visual_blocks(page_img)
        crops = crop_blocks(page_img, boxes)
        for i, crop in enumerate(crops):
            all_crops.append((p+1, i+1, crop))
    return all_crops
