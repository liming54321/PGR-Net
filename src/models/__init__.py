import os

from .builder import build_model, MODEL_REGISTRY
from easytorch.utils.registry import scan_modules

__all__ = ['build_model', 'MODEL_REGISTRY']

scan_modules(os.getcwd(), __file__, ['__init__.py', 'builder.py'])
