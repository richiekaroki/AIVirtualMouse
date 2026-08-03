"""
GPU Acceleration Support

CUDA acceleration for MediaPipe, PyTorch, and NumPy operations.
"""

import logging
import os
import platform
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import torch for CUDA detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not installed. GPU features will use CPU fallback.")


class GPUManager:
    """
    Manages GPU acceleration across the pipeline.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.cuda_available = False
        self.device = 'cpu'
        self.device_name = 'CPU'
        self.gpu_memory = 0
        self.gpu_count = 0

        self._detect_gpu()

    def _detect_gpu(self):
        """Detect available GPU resources."""
        # Check PyTorch CUDA
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.cuda_available = True
            self.device = 'cuda'
            self.gpu_count = torch.cuda.device_count()
            self.device_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"GPU detected: {self.device_name} ({self.gpu_memory:.1f} GB)")
        else:
            # Check for CUDA without PyTorch
            if self._check_cuda_available():
                self.cuda_available = True
                self.device = 'cuda'
                logger.info("CUDA available but PyTorch not using it")
            else:
                self.device = 'cpu'
                self.device_name = 'CPU'
                logger.info("Using CPU for processing")

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available without PyTorch."""
        try:
            # Check nvidia-smi
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check CUDA environment variables
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if cuda_home and os.path.exists(cuda_home):
            return True

        return False

    def get_torch_device(self):
        """Get PyTorch device for computation."""
        if not TORCH_AVAILABLE:
            return None

        if self.cuda_available:
            return torch.device('cuda')
        return torch.device('cpu')

    def optimize_mediapipe(self) -> Dict[str, Any]:
        """
        Get optimized MediaPipe configuration for current hardware.

        Returns:
            Configuration dictionary for MediaPipe
        """
        config = {
            'static_image_mode': False,
            'model_complexity': 1,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        }

        if self.cuda_available:
            # Can use higher complexity with GPU
            config['model_complexity'] = 2
            config['min_detection_confidence'] = 0.6
            config['min_tracking_confidence'] = 0.6
        else:
            # Reduce complexity for CPU
            config['model_complexity'] = 0
            config['min_detection_confidence'] = 0.4
            config['min_tracking_confidence'] = 0.4

        return config

    def get_status(self) -> Dict[str, Any]:
        """
        Get GPU status information.

        Returns:
            Status dictionary
        """
        status = {
            'device': self.device,
            'device_name': self.device_name,
            'cuda_available': self.cuda_available,
            'gpu_count': self.gpu_count,
            'gpu_memory_gb': round(self.gpu_memory, 2),
            'torch_available': TORCH_AVAILABLE
        }

        if TORCH_AVAILABLE and self.cuda_available:
            status['cuda_version'] = torch.version.cuda
            status['cudnn_version'] = torch.backends.cudnn.version()

        return status


def get_gpu_manager() -> GPUManager:
    """Get the singleton GPU manager instance."""
    return GPUManager()


def get_optimal_batch_size(model_name: str = 'default') -> int:
    """
    Get optimal batch size based on available GPU memory.

    Args:
        model_name: Name of the model to get batch size for

    Returns:
        Recommended batch size
    """
    manager = get_gpu_manager()

    if not manager.cuda_available:
        return 8  # Conservative for CPU

    # Batch sizes based on GPU memory
    if manager.gpu_memory >= 16:
        base_batch = 64
    elif manager.gpu_memory >= 8:
        base_batch = 32
    elif manager.gpu_memory >= 4:
        base_batch = 16
    else:
        base_batch = 8

    # Adjust for model size
    model_multipliers = {
        'gesture_recognizer': 0.5,
        'inference_engine': 1.0,
        'default': 1.0
    }

    multiplier = model_multipliers.get(model_name, 1.0)
    return max(4, int(base_batch * multiplier))


def get_device_info() -> str:
    """
    Get human-readable device information.

    Returns:
        String describing the processing device
    """
    manager = get_gpu_manager()

    if manager.cuda_available:
        return f"GPU: {manager.device_name} ({manager.gpu_memory:.1f} GB)"
    return "CPU: Using processor"
