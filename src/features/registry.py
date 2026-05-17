# src/features/registry.py
import importlib
import pkgutil
from .base import BaseIndicator

def discover_indicators(package_name: str = 'src.features'):
    """모든 BaseIndicator 자손을 자동 탐지."""
    indicators = {}
    package = importlib.import_module(package_name)
    
    for finder, name, ispkg in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
        try:
            mod = importlib.import_module(name)
            for attr in dir(mod):
                cls = getattr(mod, attr)
                if (isinstance(cls, type) and 
                    issubclass(cls, BaseIndicator) and 
                    cls is not BaseIndicator):
                    indicators[cls.meta.id] = cls
        except Exception:
            continue
    
    return indicators
