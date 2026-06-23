
import importlib

try:
    import requests
except ImportError:
    print("Gemini Spatial Node: 'requests' module not found. Please run 'pip install requests'")

from .gemini_spatial_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
