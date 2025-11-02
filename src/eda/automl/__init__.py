"""AutoML and advanced model selection module."""

from .model_comparison import compare_all_models, ModelComparator

# Make TPOT optional
try:
    from .tpot_models import run_tpot_optimization
    __all__ = [
        'run_tpot_optimization',
        'compare_all_models',
        'ModelComparator',
    ]
except ImportError:
    # TPOT not available, skip it
    __all__ = [
        'compare_all_models',
        'ModelComparator',
    ]
    
    def run_tpot_optimization(*args, **kwargs):
        """TPOT not installed. Install with: pip install tpot"""
        raise ImportError(
            "TPOT is not installed. Install it with: pip install tpot"
        )

