"""Dataset configuration registry."""

from .base import DatasetConfig
from .triviaqa import TriviaQAConfig
from .jeopardy import JeopardyConfig
from .bioasq import BioASQConfig

_REGISTRY = {
    "triviaqa": TriviaQAConfig,
    "jeopardy": JeopardyConfig,
    "bioasq": BioASQConfig,
}


def get_dataset_config(name: str, **kwargs) -> DatasetConfig:
    """Create a dataset configuration by name.

    Args:
        name: Dataset name (triviaqa, jeopardy, bioasq).
        **kwargs: Additional arguments passed to the config constructor
            (e.g. ranges=[(0, 5000)]).

    Returns:
        A DatasetConfig instance.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset: {name!r}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)
