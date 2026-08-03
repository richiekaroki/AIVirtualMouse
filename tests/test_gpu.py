"""Tests for GPU acceleration module."""
import pytest
from unittest.mock import patch, MagicMock
import os


class TestGPUManager:
    """Test GPUManager singleton."""

    def test_singleton_instance(self):
        from hand_motion.gpu import GPUManager
        manager1 = GPUManager()
        manager2 = GPUManager()
        assert manager1 is manager2

    def test_default_device(self):
        from hand_motion.gpu import GPUManager
        manager = GPUManager()
        assert manager.device in ['cpu', 'cuda']

    def test_get_status(self):
        from hand_motion.gpu import GPUManager
        manager = GPUManager()
        status = manager.get_status()
        assert 'device' in status
        assert 'device_name' in status
        assert 'cuda_available' in status
        assert 'torch_available' in status

    def test_optimize_mediapipe_config(self):
        from hand_motion.gpu import GPUManager
        manager = GPUManager()
        config = manager.optimize_mediapipe()
        assert 'model_complexity' in config
        assert 'min_detection_confidence' in config
        assert 'min_tracking_confidence' in config
        assert config['model_complexity'] in [0, 1, 2]


class TestGPUFunctions:
    """Test GPU utility functions."""

    def test_get_gpu_manager(self):
        from hand_motion.gpu import get_gpu_manager
        manager = get_gpu_manager()
        assert manager is not None

    def test_get_optimal_batch_size(self):
        from hand_motion.gpu import get_optimal_batch_size
        batch_size = get_optimal_batch_size()
        assert isinstance(batch_size, int)
        assert batch_size >= 4

    def test_get_optimal_batch_size_by_model(self):
        from hand_motion.gpu import get_optimal_batch_size
        for model in ['gesture_recognizer', 'inference_engine', 'default']:
            batch_size = get_optimal_batch_size(model)
            assert isinstance(batch_size, int)
            assert batch_size >= 4

    def test_get_device_info(self):
        from hand_motion.gpu import get_device_info
        info = get_device_info()
        assert isinstance(info, str)
        assert 'GPU' in info or 'CPU' in info
