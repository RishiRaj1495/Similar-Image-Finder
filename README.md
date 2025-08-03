# Similar Image Finder 🔍🖼️

## Overview

This project implements a **Similar Image Finder** system using deep learning feature extraction and fast vector search techniques. Given a query image, the system retrieves visually similar images from a dataset, enabling applications such as:

- E-commerce product recommendations
- Duplicate image detection
- Digital asset management
- Visual content discovery

The implementation utilizes the powerful **CLIP** model from OpenAI for image embeddings, combined with **FAISS** for efficient similarity search. The entire workflow is developed to run conveniently on **Google Colab**, leveraging free cloud GPUs.

---

## Features

- Preprocesses and extracts image embeddings using Hugging Face’s CLIP model (`clip-vit-base-patch16`)
- Indexes embeddings with FAISS to enable fast cosine similarity search
- Supports querying with a new/uploaded image to find top-N most similar images
- Visualizes query image and retrieved similar images side-by-side
- Easily extendable to different datasets or larger-scale deployments

---

## Getting Started

### Prerequisites

- Python 3.x (Google Colab recommended)
- Libraries installed in notebook:
  - torch, torchvision, transformers, pillow, faiss-cpu, matplotlib, numpy

These libraries are installed automatically in the Colab notebook with pip commands.

### Dataset Preparation

- Upload your own images into a folder named `images/` within Colab
- Or download sample image URLs programmatically in the notebook

### Running the Notebook

1. Run all cells sequentially to:
   - Install dependencies
   - Load and preprocess images
   - Extract image features using the CLIP model
   - Build FAISS index for similarity search
   - Query with a sample image and retrieve top-N similar images
   - Visualize the results inline

2. Update image paths dynamically using Python to handle any number of images.

### Querying & Results

- Use the `get_image_features()` function to generate embeddings for any query image
- Search the FAISS index to find the closest matches
- Display the query and matched images with matplotlib

---

## How to Update Image Paths

Instead of hardcoding file names, dynamically list all supported image files in your dataset folder using this snippet:

import os

image_folder = "images"
image_paths = [os.path.join(image_folder, fname)
for fname in os.listdir(image_folder)
if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(image_paths)} images.")

---

## Results Visualization

After searching, visualize the query and its top similar images:


---

## Project Structure

- `Similar_Image_Finder.ipynb` — Full Google Colab notebook with code, explanations, and visualizations
- `images/` — Folder containing dataset images (user uploaded/downloaded)
- `README.md` — This documentation file

---

## Future Enhancements

- Support for text queries leveraging CLIP’s multimodal capability
- Scalable index using FAISS on very large image datasets
- Building a user-friendly web interface with Streamlit or Gradio
- Model fine-tuning to domain-specific image collections
- Deployment as a REST API for production use

---

## References

- CLIP Model: https://huggingface.co/openai/clip-vit-base-patch16
- Google Colab Platform: https://colab.research.google.com/

---

## Contact & Support

Feel free to open issues or submit pull requests for improvements!


---

🚀 Thank you for checking out the Similar Image Finder project! 
