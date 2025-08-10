import logging
from PIL import Image
from pdf2image import convert_from_path
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import io

logger = logging.getLogger(__name__)

def extract_images_from_pdf(pdf_path):
    """
    Converts each PDF page to a single image. This is a reliable method.
    """
    images = []
    metadata_list = []
    try:
        # Find poppler path for robustness on different systems
        possible_bins = ["/opt/homebrew/bin", "/usr/local/bin"]
        poppler_bin = next((p for p in possible_bins if os.path.exists(os.path.join(p, "pdftoppm"))), None)
        
        page_images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_bin) if poppler_bin else convert_from_path(pdf_path, dpi=300)
        
        for i, page_image in enumerate(page_images):
            images.append(page_image)
            metadata_list.append({
                "page_number": i + 1,
                "extracted_label": f"Page {i + 1}"
            })
        
        logger.info(f"Converted {len(page_images)} PDF pages to images.")
        source_metadata = {"source": pdf_path}
        return images, source_metadata, metadata_list
        
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}", exc_info=True)
        return [], {}, []

def extract_images_from_url(url, max_images=50):
    """
    Extracts images from any URL by scraping <img> tags with JPG/PNG sources.
    Optimized for medical sites like PathologyOutlines.com.
    """
    images = []
    metadata_list = []
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        logger.info(f"Fetching page: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        img_tags = soup.find_all('img')
        logger.info(f"Found {len(img_tags)} potential images. Filtering and downloading...")
        
        for section_index, img_tag in enumerate(img_tags, start=1):
            img_url = img_tag.get('src')
            if not img_url or not img_url.lower().endswith(('.jpg', '.png', '.jpeg')) or 'logo' in img_url.lower() or 'icon' in img_url.lower():
                continue  # Skip logos, icons, or non-medical images
            
            full_img_url = urljoin(url, img_url)
            
            try:
                img_response = requests.get(full_img_url, timeout=10)
                img_response.raise_for_status()
                
                pil_image = Image.open(io.BytesIO(img_response.content))
                # Skip very small images (e.g., <100x100 pixels) to avoid thumbnails/icons
                if pil_image.width < 100 or pil_image.height < 100:
                    continue
                
                images.append(pil_image)
                
                extracted_label = img_tag.get('alt') or f"Image_{section_index}"
                metadata_list.append({
                    "section_index": section_index,
                    "extracted_label": extracted_label
                })
                
                if len(images) >= max_images:
                    break
            except Exception as e:
                logger.warning(f"Failed to download or open image from {full_img_url}: {e}")
        
        source_metadata = {"source": url}
        logger.info(f"Successfully extracted {len(images)} images from URL.")
        return images, source_metadata, metadata_list
    
    except requests.RequestException as e:
        logger.error(f"Failed to fetch URL {url}: {e}", exc_info=True)
        return [], {}, []
