import logging
import torch
import numpy as np
from cellpose import models
from typing import Tuple, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BioSegmentor:
    def __init__(self, model_type: str = 'cyto', use_gpu: bool = True):
        """
        Initialize the Cellpose segmentor.

        Args:
            model_type (str): 'cyto', 'nuclei', or 'cyto2'. 
                              'cyto' is good for general cells.
            use_gpu (bool): Whether to use GPU acceleration.
        """
        self.device = self._get_device(use_gpu)
        logger.info(
            f"Initializing Cellpose model '{model_type}' on device: {self.device}")

        # Cellpose handles 'gpu' bool or specific device handling internally,
        # but for MPS specifically we might need to ensure torch sees it.
        # fast check:
        gpu_flag = False
        if self.device.type in ['cuda', 'mps']:
            gpu_flag = True

        self.model = models.CellposeModel(
            gpu=gpu_flag, model_type=model_type, device=self.device)

    def _get_device(self, use_gpu: bool) -> torch.device:
        if not use_gpu:
            return torch.device('cpu')

        if torch.backends.mps.is_available():
            logger.info("Apple Silicon (MPS) detected.")
            return torch.device('mps')
        elif torch.cuda.is_available():
            logger.info("CUDA GPU detected.")
            return torch.device('cuda')
        else:
            logger.warning("No GPU detected, falling back to CPU.")
            return torch.device('cpu')

    def segment(self,
                image: np.ndarray,
                channels: List[int] = [0, 0],
                diameter: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Run segmentation on the provided image.

        Args:
            image (np.ndarray): Input image, typically (Y, X) or (Y, X, C).
                                Cellpose expects (Y, X) for single channel or (Y, X, C) for multi.
                                NOTE: If your image is (C, Y, X), you must transpose it before passing here!
            channels (List[int]): [cytoplasm, nucleus]. 
                                  0=grayscale, 1=red, 2=green, 3=blue.
                                  Or if channels are indices in a multi-channel image, pass relevant indices if properly responding to Cellpose specs.
                                  For Bio-Oracle, if inputs are already specific arrays, careful handling is needed.
                                  Cellpose docs: channels=[0,0] means grayscale.
            diameter (float, optional): Estimated cell diameter. If None, cellpose estimates it.

        Returns:
            Tuple[np.ndarray, float]: 
                - masks: Array of same shape as input (Y, X) with integer labels 1..N.
                - flow: (Optional) Flow fields, here simplified to just diameter for this implementation return or strictly masks.
        """
        logger.info(f"Starting segmentation. Image shape: {image.shape}")

        try:
            # Run inference
            # Run inference
            # CellposeModel.eval returns 3 values: masks, flows, styles
            masks, flows, styles = self.model.eval(
                image,
                diameter=diameter,
                channels=channels,
                flow_threshold=0.4,
                # Assuming 2D for now as per Phase 1 description (implied)
                do_3D=False
            )

            # Since CellposeModel doesn't return estimated diameter, we use the provided one
            # or None if not provided.
            diams = diameter if diameter is not None else 30.0  # Default fallback

            logger.info(
                f"Segmentation complete. Found {masks.max()} cells. Diameter: {diams}")
            return masks, diams

        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            raise e


# Helper function
def run_segmentation_workflow(image_data: np.ndarray, channels_config: List[int]):
    """
    Wrapper for easier calling from the pipeline.
    Expects data in proper format for Cellpose.
    """
    segmentor = BioSegmentor()
    masks, _ = segmentor.segment(image_data, channels=channels_config)
    return masks
