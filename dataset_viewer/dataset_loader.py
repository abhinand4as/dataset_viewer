# --- dataset_viewer/dataset_loader.py ---
# This file contains the main logic, encapsulated in a class.

import os
import xml.etree.ElementTree as ET
import logging
import yaml

import fiftyone as fo
import fiftyone.types as fot

class GenericDatasetViewer:
    """
    A general tool to view a dataset in FiftyOne with annotations.

    This class loads a dataset from a directory, parses the corresponding
    annotation files (currently VOC-style XML), and attaches the
    segmentation data to the FiftyOne dataset for visualization.
    """
    def __init__(self, config_path):
        """
        Initializes the GenericDatasetViewer with a configuration file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        self.dataset_name = self.config.get("dataset_name")
        self.image_dir = self.config.get("image_dir")
        self.annotation_dir = self.config.get("annotation_dir")
        self.annotation_format = self.config.get("annotation_format")
        self.coco_annotation_file = self.config.get("coco_annotation_file")

        if not self.dataset_name or not self.image_dir or not self.annotation_dir:
            self.logger.error("Configuration file is missing required fields.")
            raise ValueError("Configuration file must contain 'dataset_name', 'image_dir', and 'annotation_dir'")

        self.logger.info(f"Initialized GenericDatasetViewer for dataset: {self.dataset_name}")
        self.logger.info(f"Image directory: {self.image_dir}")
        self.logger.info(f"Annotation directory: {self.annotation_dir}")

    def _load_config(self, config_path):
        """Loads configuration from a YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config file: {e}")

    def _setup_logging(self):
        """Configures the logging system for the tool."""
        log_file = self.config.get("log_file", "dataset_viewer.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("GenericDatasetViewer")

    def _load_dataset(self):
        """Loads images into a FiftyOne dataset."""
        self.logger.info("Loading images into FiftyOne dataset...")
        try:
            if self.annotation_format == "VOC":
                dataset = fo.Dataset.from_dir(
                    dataset_dir=self.image_dir,
                    dataset_type=fot.ImageDirectory,
                    name=self.dataset_name,
                    persistent=True, # Persist the dataset to avoid re-loading from scratch
                )
                self.logger.info(f"Successfully loaded dataset with {len(dataset)} samples.")
                return dataset
            elif self.annotation_format == "COCO":
                dataset = fo.Dataset.from_dir(
                    dataset_type=fo.types.COCODetectionDataset,
                    data_path=self.image_dir,
                    labels_path=self.coco_annotation_file,
                    name=self.dataset_name,
                    persistent=True,
                )  
                self.logger.info(f"Successfully loaded dataset with {len(dataset)} samples.")
                return dataset 
        except Exception as e:
            self.logger.error(f"Failed to load dataset from {self.image_dir}: {e}")
            return None

    def _process_annotations(self, dataset):
        """
        Parses VOC XML annotations and attaches them to the dataset.
        NOTE: This method is currently hard-coded for VOC XML format.
        """
        self.logger.info("Processing annotations and attaching to dataset...")
        
        # Loop over each sample in the dataset
        for sample in dataset.iter_samples(autosave=True):
            # Match XML annotation file by image filename (without extension)
            sample_root = os.path.splitext(os.path.basename(sample.filepath))[0]
            xml_file = os.path.join(self.annotation_dir, sample_root + ".xml")

            if not os.path.exists(xml_file):
                self.logger.warning(f"No annotation file found for {sample_root}.xml. Skipping...")
                continue
            
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Get image dimensions from XML
                width_elem = root.find("size/width")
                height_elem = root.find("size/height")
                
                if width_elem is None or height_elem is None:
                    self.logger.warning(f"Image dimensions not found in {xml_file}. Skipping...")
                    continue
                
                width = float(width_elem.text)
                height = float(height_elem.text)

                polylines_list = []

                # Iterate over objects in XML
                for obj in root.findall("object"):
                    label_elem = obj.find("name")
                    polygon_elem = obj.find("polygon")
                    
                    if label_elem is None or polygon_elem is None:
                        self.logger.warning(f"Object or polygon element not found in {xml_file}. Skipping object...")
                        continue

                    label = label_elem.text.strip()
                    
                    # Extract ordered polygon points
                    points = []
                    i = 1
                    while True:
                        x_elem = polygon_elem.find(f"x{i}")
                        y_elem = polygon_elem.find(f"y{i}")
                        if x_elem is None or y_elem is None:
                            break

                        # Normalize to [0,1]
                        x_rel = float(x_elem.text) / width
                        y_rel = float(y_elem.text) / height
                        points.append((x_rel, y_rel))
                        i += 1
                
                    if len(points) >= 3:
                        polyline = fo.Polyline(
                            label=label,
                            points=[points],
                            closed=True,
                            filled=True,
                        )
                        polylines_list.append(polyline)

                # Attach polylines to the sample
                if polylines_list:
                    sample["segmentations"] = fo.Polylines(polylines=polylines_list)
                    sample.save()
                    self.logger.info(f"Annotations for {sample_root}.xml successfully attached.")

            except (ET.ParseError, ValueError, AttributeError) as e:
                self.logger.error(f"Failed to parse or process {xml_file}: {e}")
                
        self.logger.info("Finished processing annotations.")

    def run(self):
        """
        Runs the full process of loading a dataset, processing annotations,
        and launching the FiftyOne app.
        """
        dataset = self._load_dataset()
        if dataset:
            if self.annotation_format == "VOC":
                self._process_annotations(dataset)
                self.logger.info("Launching FiftyOne app...")
                session = fo.launch_app(dataset)
                session.wait()
                self.logger.info("FiftyOne app session closed.")
            elif self.annotation_format == "COCO":
                self.logger.info("Launching FiftyOne app...")
                session = fo.launch_app(dataset)
                session.wait()
                self.logger.info("FiftyOne app session closed.")