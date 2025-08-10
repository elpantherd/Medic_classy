# Medic_classy
Zero shot medical vs non-medical image classifier

# Medical Image Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CLIP Model](https://img.shields.io/badge/Model-CLIP%20(ViT--B/32)-orange)](https://github.com/openai/CLIP)

A robust, zero-shot image classification tool that distinguishes between medical and non-medical images from PDFs or websites. Built using OpenAI's CLIP model for efficient, training-free classification. This project demonstrates GenAI engineering principles, including multimodal processing, prompt engineering, and modular design.

------

## Problem Understanding
The task requires extracting and classifying images from PDFs or URLs as medical (e.g., scans, pathology slides) or non-medical (e.g., diagrams, logos). This is a GenAI challenge involving computer vision, web scraping, and multimodal models.
## Core Approach
 - Image Extraction:
 - PDFs: Use pdf2image to convert each page to an image (reliable for medical documents).
 - URLs: Scrape with BeautifulSoup, filtering for JPG/PNG sources and skipping small icons/logos (optimized for medical sites like PathologyOutlines.com).
 - Classification: Employ zero-shot learning with CLIP (ViT-B/32) to match images against text prompts (e.g., “a diagnostic medical image like an MRI or pathology slide” vs. “a non-medical diagram”).
 - Custom prompts for input types: General for PDFs, website-adapted for scraped content.
 - Output: JSON with per-image labels, confidences, and summaries (e.g., lists of medical pages/sections).
 - Tech Stack: Python, Torch, open_clip, pdf2image, requests, BeautifulSoup. No custom training—leverages CLIP’s pre-trained capabilities.

 - Why CLIP? Zero-shot eliminates data labeling needs, crucial for medical domains with privacy constraints. It’s efficient (runs on CPU/MPS) and handles diverse images via prompts. Alternatives like fine-tuned ResNet were rejected due to training overhead.
 - Ethical Considerations: Avoids sensitive data; prompts are tuned to reduce bias (e.g., including “pathology slide” for microscopic views).
 - Modularity: Code is separated (e.g., `data_loader.py` for extraction, `predictor.py` for classification) for maintainability.

---

## Project Overview

This application classifies images as **medical** or **non-medical** using CLIP (Contrastive Language-Image Pretraining), a vision-language model. It supports:

- **PDF Inputs**: Extracts and classifies images from each PDF page (e.g., medical reports, scans).
- **Website Inputs**: Scrapes and classifies images from URLs (e.g., medical image gallery sites like PathologyOutlines.com).

### Key Features

- Zero-shot classification (no training required).
- Custom prompts adapted for PDFs vs. websites to boost accuracy.
- JSON output with individual classifications and summarized results (e.g., pages or sections marked medical/non-medical).
- Efficient image preprocessing and extraction.
- Modular and reproducible codebase suited for GenAI engineering assessments.

---

## Model Selection Rationale

I selected OpenAI's **CLIP (ViT-B/32)** for these reasons:

- **Zero-Shot Learning**: Eliminates the need for labeled datasets, a major challenge in medical AI due to privacy and scarcity.
- **Multimodal Capability**: Understands and matches images with natural language prompts.
- **Efficiency**: Lightweight and compatible with CPU, GPU, and Apple MPS for performance on various devices.
- **Ethical AI Considerations**: No fine-tuning means reduced risk of overfitting or bias leakage from medical data.
- **Prompt Engineering**: Custom textual prompts specific to input source further improve classification accuracy.

Alternatives like CNNs or fine-tuned models require extensive labeled data and training time, impractical for rapid development and assessment.

## Accuracy Results on a Small Validation/Test Set

I evaluated on two test cases from the attached JSON files (20 PDF images + 30 website images = 50 total). Ground truth was manually assigned based on content (e.g., all pathology slides as medical, logos as non-medical).

- **PDF Test Set (20 images from Document 1.pdf):**
  - Ground Truth: 11 medical (e.g., scans), 9 non-medical (e.g., diagrams).
  - Model Results: 11 medical, 9 non-medical (100% match).
  - Accuracy: 100% (all correct; average confidence 0.92 for medical, 0.88 for non-medical).
  - **Notes:** High performance on mixed content; no misclassifications.

- **Website Test Set (30 images from PathologyOutlines DCIS page):**
  - Ground Truth: 27 medical (e.g., histology/tissue images), 3 non-medical (e.g., missing placeholders, icons).
  - Model Results: 27 medical, 3 non-medical (100% match).
  - Accuracy: 100% (average confidence 0.98 for medical, 0.95 for non-medical).
  - **Notes:** Excellent on pathology-focused site; prompts handled microscopic views well.

**Overall Accuracy:** 100% across 50 images.  
**Limitations:** Small set; real-world accuracy may vary with image quality.  
**Future Work:** Expand dataset for robust statistical evaluation.
**Performance/Efficiency Considerations**
	- Runtime: PDF (20 images): ~2 seconds (9.93 img/sec on MPS). Website (30 images): ~1.4 seconds (20.8 img/sec). Scalable to larger inputs.
	- Resource Use: CLIP is lightweight (~63M parameters); uses ~500MB RAM on CPU. MPS acceleration boosts speed 2-3x on Apple Silicon.
	- Efficiency Optimizations: Single-image processing (no batching needed for zero-shot); size filters skip irrelevant images. Potential improvements: Parallel scraping or GPU batching for 100+ images.
	- Trade-offs: Zero-shot is fast but may have lower accuracy on ambiguous images vs. fine-tuned models (which require more compute/data).


 
---

## Setup Guide

### Prerequisites (All Platforms)

- Install Git: [https://git-scm.com/downloads](https://git-scm.com/downloads)
- Clone the repo:
``git clone https://github.com/elpantherd/Medic_classy.git
   cd medical-image-classifier``

- Create and activate a virtual environment:
- Linux/macOS:
  ```
  python3 -m venv venv
  source venv/bin/activate
  ```
- Windows (PowerShell):
  ```
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- Install dependencies:
```
pip install -r requirements.txt
```
- Install Poppler (required for PDF page extraction):
- macOS:
  ```
  brew install poppler
  ```
- Ubuntu/Debian:
  ```
  sudo apt update
  sudo apt install poppler-utils libjpeg-dev zlib1g-dev libopenjp2-7
  ```
- Windows:
  - Install via Chocolatey or download from [http://blog.alivate.com.au/poppler-windows/](http://blog.alivate.com.au/poppler-windows/)
  - Ensure Poppler's `bin` folder is added to your system PATH.

---

### Platform Notes

| OS       | Notes                                                                  |
|----------|------------------------------------------------------------------------|
| macOS    | Supports MPS acceleration on Apple Silicon machines.                   |
| Linux    | CUDA setup recommended if NVIDIA GPU available.                       |
| Windows  | Requires proper environment setup for Poppler and GPU CUDA drivers.    |

---

## Usage

Run the main script from your activated environment.

- **Classify images from a PDF file:**
```
python main.py “https://www.pathologyoutlines.com/topic/breastmalignantdcis.html” –output_dir “output/” –device mps
```
- **Classify images from a website:**
```
python main.py "/path/to/your.pdf" --output_dir "output/" --device cpu
```
(replace `cpu` with `mps` or `cuda` if available).

## Code Structure

- `main.py`: CLI entry point, orchestrates extraction and classification.
- `src/data_loader.py`: Handles PDF page conversion and web image scraping.
- `src/model.py`: Defines the CLIP-based zero-shot classifier.
- `src/predictor.py`: Runs classification with input-specific prompts.
- `src/utils.py`: Logging setup and helpers.

---

## License

MIT License — see [LICENSE](LICENSE) for full terms.

---

## Contact

For questions or suggestions, please open an issue or contact at [dthayalan760@gmail.com].

---

