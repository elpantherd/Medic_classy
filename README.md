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


| Feature               | CLIP ViT-B/32 | Fine-Tuned CNNs | Custom Transformer |
|-----------------------|---------------|-----------------|--------------------|
| Zero-Shot Capability  | ✅ Yes         | ❌ No            | ❌ No              |
| Medical Privacy Safe  | ✅ Yes         | ⚠ Dataset risk  | ⚠ Dataset risk    |
| Inference Speed       | ✅ Fast        | ⚠ Moderate      | ❌ Slow           |
| Hardware Flexibility  | ✅ CPU/MPS/GPU | ⚠ GPU only      | ⚠ GPU only       |

CLIP allows **text–image matching without supervision**, avoiding dataset creation while retaining high domain generalization.


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
- **Resource Use**: CLIP ViT-B/32 is lightweight, using ~500MB RAM on CPU. MPS acceleration provides a 2-3x speedup on Apple Silicon.
- **Optimizations**: Processing is done per-image (no batching needed for zero-shot). Size filters are used to skip irrelevant small images.
- **Trade-offs**: Zero-shot is extremely fast but may have lower accuracy on ambiguous images compared to fine-tuned models, which require significantly more data and compute resources.

**Test Set:** 50 images (20 from PDFs, 30 from web)  

| Source | Images | GT Medical | GT Non-Medical | Accuracy | Avg Conf. (Med) | Avg Conf. (Non-Med) |
|--------|--------|-----------|----------------|----------|-----------------|---------------------|
| PDF    | 20     | 11        | 9              | 100%     | 0.92            | 0.88                |
| Web    | 30     | 27        | 3              | 100%     | 0.98            | 0.95                |
| **Total** | **50** | **38**   | **12**         | **100%** | —               | —                   |

**Runtime:**  
- PDF: 20 images in ~2 sec (~9.9 img/sec, MPS)  
- Web: 30 images in ~1.4 sec (~20.8 img/sec, MPS)  

**Memory Use:** ~500 MB on CPU, 2–3× faster with Apple MPS  

---

## ⚙️ Setup Guide

### 1. Prerequisites
- First, ensure you have Git installed from [git-scm.com/downloads](https://git-scm.com/downloads).
- Clone the repository by running `git clone https://github.com/elpantherd/Medic_classy.git` and navigate into the new directory with `cd Medic_classy`.
- Create a virtual environment by running `python3 -m venv venv`.
- Activate the environment. On **Linux or macOS**, run `source venv/bin/activate`. On **Windows PowerShell**, run `.\venv\Scripts\Activate.ps1`.
- Finally, install all required packages by running `pip install -r requirements.txt`.

### 2. Install Poppler
`Poppler` is a system dependency required for handling PDFs.
- On **macOS**, you can install it via Homebrew by running `brew install poppler`.
- On **Ubuntu/Debian** Linux systems, run `sudo apt-get update && sudo apt-get install -y poppler-utils`.
- On **Windows**, you must download Poppler manually. Go to [this repository](https://github.com/oschwartz10612/poppler-windows/releases/), download the latest release, extract the archive, and add the full path to its `bin` folder to your system's PATH environment variable.

### Platform Notes

| OS      | Notes                                                              |
| :------ | :----------------------------------------------------------------- |
| **macOS** | Natively supports MPS acceleration on Apple Silicon (M1/M2/M3).    |
| **Linux** | CUDA is recommended for acceleration if an NVIDIA GPU is available. |
| **Windows** | Requires careful setup of Poppler PATH and GPU (CUDA) drivers.    |

---

## ▶️ Usage
Run the main script from your activated virtual environment.

- **Classify images from a PDF file:**
  ```bash
  python main.py "/path/to/your/document.pdf" --output_dir "output/" --device cpu
  ```
- **Classify images from website**
```bash
python main.py "https://www.pathologyoutlines.com/topic/breastmalignantdcis.html" --output_dir "output/" --device mps
```
Replace --device with cpu, mps, or cuda based on your hardware.

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
