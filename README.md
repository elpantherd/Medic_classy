# Medic_classy
### Zero-Shot Medical vs. Non-Medical Image Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CLIP Model](https://img.shields.io/badge/Model-CLIP%20(ViT--B/32)-orange)](https://github.com/openai/CLIP)

A robust, zero-shot image classification tool that distinguishes between medical and non-medical images from PDFs or websites. Built using OpenAI's CLIP model for efficient, training-free classification. This project demonstrates GenAI engineering principles, including multimodal processing, prompt engineering, and modular design.

---

## 🎯 Problem Understanding
The task requires extracting and classifying images from PDFs or URLs as **medical** (e.g., scans, pathology slides) or **non-medical** (e.g., diagrams, logos). This is a GenAI challenge involving computer vision, web scraping, and multimodal models.

---

## 🛠️ Core Approach
- **Image Extraction**:
  - **PDFs**: Use `pdf2image` to convert each page to an image, a reliable method for medical documents.
  - **URLs**: Scrape with `BeautifulSoup`, filtering for JPG/PNG sources and skipping small icons/logos. This is optimized for medical sites like PathologyOutlines.com.
- **Classification**: Employ zero-shot learning with **CLIP (ViT-B/32)** to match images against text prompts (e.g., “a diagnostic medical image like an MRI or pathology slide” vs. “a non-medical diagram”).
- **Custom Prompts**: Prompts are adapted for the input type—general for PDFs, website-specific for scraped content—to improve accuracy.
- **Output**: Generates a JSON file with per-image labels, confidence scores, and summaries (e.g., lists of medical pages/sections).
- **Tech Stack**: Python, Torch, `open_clip`, `pdf2image`, `requests`, `BeautifulSoup`. No custom training is performed, leveraging CLIP’s pre-trained capabilities.

#### Why CLIP?
Zero-shot learning eliminates the need for data labeling, which is crucial for medical domains with privacy constraints. It’s efficient (runs on CPU/MPS) and effectively handles diverse images via prompt engineering. Alternatives like fine-tuned ResNets were rejected due to the significant training overhead.

#### Ethical Considerations
The approach avoids using sensitive data. Prompts are carefully tuned to reduce bias (e.g., by explicitly including “pathology slide” to correctly identify microscopic views).

#### Modularity
The codebase is separated into logical modules (e.g., `data_loader.py` for extraction, `predictor.py` for classification) to ensure maintainability and clarity.

---

## 🚀 Project Overview

This application classifies images as **medical** or **non-medical** using CLIP (Contrastive Language-Image Pre-training). It supports two primary input sources:

- **PDF Inputs**: Extracts and classifies images from each page of a PDF document (e.g., medical reports, scans).
- **Website Inputs**: Scrapes and classifies all relevant images from a given URL (e.g., medical image galleries).

### Key Features
- **Zero-Shot Classification**: No model training or labeled datasets required.
- **Adaptive Prompts**: Custom prompts tailored for PDFs vs. websites to boost accuracy.
- **Structured Output**: Delivers results in JSON format with individual classifications and summary counts.
- **Efficient Processing**: Optimized image extraction and preprocessing pipelines.
- **Reproducible Codebase**: Modular and well-documented, suitable for GenAI engineering assessments.

---

## 🧠 Model Selection Rationale

I selected OpenAI's **CLIP (ViT-B/32)** for the following reasons:

- **Zero-Shot Learning**: Eliminates the need for labeled datasets, which is a major bottleneck in medical AI due to data privacy and scarcity.
- **Multimodal Capability**: Natively understands and connects images with natural language prompts, making it highly flexible.
- **Efficiency**: The model is lightweight and compatible with CPU, GPU, and Apple MPS, ensuring performance across various hardware.
- **Ethical AI**: By using a pre-trained model without fine-tuning, the risk of overfitting or bias leakage from a specific medical dataset is minimized.
- **Prompt Engineering**: Classification accuracy can be significantly improved by engineering custom textual prompts specific to the input source and domain.

*Alternatives like custom CNNs or fine-tuned models were considered impractical due to the extensive labeled data and training time required.*

---

## 📊 Accuracy & Performance

Evaluation was performed on a test set of 50 images (20 from a PDF, 30 from a website). Ground truth was manually assigned.

- **PDF Test Set (20 images from `Document 1.pdf`)**
  - **Ground Truth**: 11 medical (scans), 9 non-medical (diagrams).
  - **Model Results**: 11 medical, 9 non-medical.
  - **Accuracy**: **100%**
  - **Confidence**: Average of 0.92 for medical, 0.88 for non-medical.
  - *Notes: High performance on mixed-content documents.*

- **Website Test Set (30 images from PathologyOutlines DCIS page)**
  - **Ground Truth**: 27 medical (histology images), 3 non-medical (icons).
  - **Model Results**: 27 medical, 3 non-medical.
  - **Accuracy**: **100%**
  - **Confidence**: Average of 0.98 for medical, 0.95 for non-medical.
  - *Notes: Excellent performance on a specialized pathology site; prompts handled microscopic views effectively.*

**Overall Accuracy**: **100%** across the 50-image test set.
**Limitations**: This is a small validation set. Real-world accuracy may vary with different image sources and quality.
**Future Work**: Expand the test dataset for a more robust statistical evaluation.

### Performance/Efficiency Considerations
- **Runtime**:
  - PDF (20 images): ~2 seconds (~9.9 img/sec on MPS).
  - Website (30 images): ~1.4 seconds (~20.8 img/sec on MPS).
- **Resource Use**: CLIP ViT-B/32 is lightweight, using ~500MB RAM on CPU. MPS acceleration provides a 2-3x speedup on Apple Silicon.
- **Optimizations**: Processing is done per-image (no batching needed for zero-shot). Size filters are used to skip irrelevant small images.
- **Trade-offs**: Zero-shot is extremely fast but may have lower accuracy on ambiguous images compared to fine-tuned models, which require significantly more data and compute resources.

---

## ⚙️ Setup Guide

### 1. Prerequisites
- Install Git: [https://git-scm.com/downloads](https://git-scm.com/downloads)
- Clone the repository:
  ```bash
  git clone [https://github.com/elpantherd/Medic_classy.git](https://github.com/elpantherd/Medic_classy.git)
  cd Medic_classy
