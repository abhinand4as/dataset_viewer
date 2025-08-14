# --- run.py ---
# This is the entry point for the tool.

import os
from dataset_viewer.dataset_loader import GenericDatasetViewer

def main():
    """Entry point for the application."""
    # Assume config.yaml is in the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "dataset_viewer", "config.yaml")
    
    try:
        viewer = GenericDatasetViewer(config_path)
        viewer.run()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        # Exit gracefully if the tool cannot be initialized
        return

if __name__ == "__main__":
    main()