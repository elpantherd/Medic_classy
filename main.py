import argparse
import os
import json
import logging
from datetime import datetime
from src.predictor import ImageClassifier
from src.data_loader import extract_images_from_pdf, extract_images_from_url
from src.utils import setup_logging

def main(args):
    """
    Main function to orchestrate the image classification pipeline for both PDFs and URLs.
    """
    logger = logging.getLogger(__name__)
    
    # Expand user paths (e.g., ~) for robustness
    input_path = os.path.expanduser(args.input_path)
    is_url = input_path.lower().startswith(('http://', 'https://'))
    is_pdf = input_path.lower().endswith('.pdf')

    # 1. Extract Images based on input type
    logger.info(f"Starting image extraction from: {input_path}")
    if is_url:
        images, source_metadata, metadata_list = extract_images_from_url(input_path)
    elif is_pdf:
        images, source_metadata, metadata_list = extract_images_from_pdf(input_path)
    else:
        logger.error("Unsupported input type. Please provide a URL or a .pdf file path.")
        return

    if not images:
        logger.warning("No images found in the provided source.")
        return

    logger.info(f"Successfully extracted {len(images)} images.")

    # 2. Initialize Classifier and Predict (pass is_url flag)
    try:
        classifier = ImageClassifier(device=args.device, batch_size=args.batch_size, is_url=is_url)
        logger.info("Classifier initialized. Starting prediction...")
        results = classifier.classify_images(images, source_metadata['source'], metadata_list)
        logger.info("Prediction complete.")
    except Exception as e:
        logger.error(f"An error occurred during classification: {e}", exc_info=True)
        return

    # 3. Add Collective Summary to the Results
    classifications = results.get("classifications", [])
    
    if is_pdf:
        medical_pages = sorted(list(set([c["page_number"] for c in classifications if c.get("classification") == "medical"])))
        non_medical_pages = sorted(list(set([c["page_number"] for c in classifications if c.get("classification") == "non-medical"])))
        results["summary"] = {
            "medical_pages": medical_pages,
            "non_medical_pages": non_medical_pages
        }
    elif is_url:
        medical_sections = sorted(list(set([c["section_index"] for c in classifications if c.get("classification") == "medical"])))
        non_medical_sections = sorted(list(set([c["section_index"] for c in classifications if c.get("classification") == "non-medical"])))
        results["summary"] = {
            "medical_sections": medical_sections,
            "non_medical_sections": non_medical_sections
        }

    # 4. Save Final Results to JSON
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize filename from URL
    if is_url:
        output_filename = input_path.replace('https://', '').replace('http://', '').replace('/', '_').replace('?', '_').replace('=', '_')
    else:
        output_filename = os.path.basename(input_path)

    # Truncate long filenames
    if len(output_filename) > 50:
        output_filename = output_filename[:50]

    output_filepath = os.path.join(args.output_dir, f"results_{output_filename}_{timestamp}.json")

    try:
        with open(output_filepath, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {output_filepath}")
        print(f"\n✅ Success! Results saved to: {output_filepath}")
    except IOError as e:
        logger.error(f"Failed to write results to file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify images from PDFs or URLs as medical or non-medical.")
    parser.add_argument("input_path", type=str, help="File path to a PDF or a URL of a web page.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the JSON results.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for model inference (Note: CLIP processes one by one).")
    parser.add_argument("--device", type=str, default=None, help="Device to use for computation ('cuda', 'mps', or 'cpu').")
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = os.path.join(args.output_dir, 'logs')
    setup_logging(log_dir)
    
    main(args)
