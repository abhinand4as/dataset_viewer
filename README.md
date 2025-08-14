# Dataset Viewer

`dataset-viewer` is a Python-based tool for visualizing datasets with annotations using [FiftyOne](https://voxel51.com/). It supports VOC-style XML annotations and provides an easy way to load, process, and visualize datasets.

## Features

- Load datasets from a directory of images.
- Parse VOC-style XML annotations and attach them to the dataset.
- Visualize datasets and annotations using FiftyOne's interactive app.

## Requirements

- Python 3.10.17 or higher
- Dependencies listed in `pyproject.toml`:
  - `fiftyone>=1.7.2`
  - `opencv-python>=4.12.0.88`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd dataset_viewer
2. Set up a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
3. Install dependencies using uv:
    ```bash
    uv sync
    ```
4. Configure the dataset paths in `dataset_viewer/config.yaml`:
   ```yaml
   dataset_name: your_dataset_name
   image_dir: /path/to/images
   annotation_dir: /path/to/annotations
   log_file: dataset_viewer.log
   ```
## Usage
 Run the application:
   ```bash
   uv run run.py
   ```
