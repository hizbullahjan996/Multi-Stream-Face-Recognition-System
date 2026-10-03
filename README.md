Multi-Stream Face Recognition System

A GPU-accelerated real-time multi-stream face recognition pipeline using InsightFace's SCRFD face detector and W600K face embedding model, optimized with TensorRT and NVDEC hardware decoding (NVIDIA GPU required).


## Overview

This system processes up to 4 video streams simultaneously for face detection, recognition, and tracking using TensorRT-accelerated inference on NVIDIA GPUs.


## Pipeline

Stream 1..4 ? NVDEC GPU Decode ? SCRFD Detection (TensorRT) ? Face Crop ? W600K Embedding (TensorRT) ? Gallery DB ? Cosine Similarity (>=0.60 = Known) ? Annotated MP4


## Features

- GPU-accelerated with TensorRT + NVDEC hardware decoding (NVIDIA GPU required)
- Processes up to 4 concurrent video streams
- Persistent incremental gallery with content-hash deduplication
- Frame chunking and batching for optimal TensorRT performance
- IoU tracking with recognition caching
- Multiple processing modes (High Accuracy/Balanced/High Speed)
- Cosine similarity threshold: 0.60 (60%)


## Files

- face_recognition.ipynb - Complete Colab notebook implementation
- Face_Recognition_System_Presentation.pptx - Project presentation
- README.md - Documentation


## Usage

Open face_recognition.ipynb in Google Colab with GPU runtime, mount Drive, and run cells sequentially to build engines, upload gallery, and process videos.
"# Multi-Stream-Face-Recognition-System" 
