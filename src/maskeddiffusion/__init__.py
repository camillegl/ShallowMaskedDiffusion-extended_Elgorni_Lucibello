"""Hidden-manifold masked diffusion: clean, tested implementation.

Scientific contract: docs/RESEARCH_SPEC.md and docs/NOTATION.md.
"""

__version__ = "0.2.0"

from .dimensions import Dimensions
from .randomness import SeedHierarchy
from .teacher import HiddenManifoldTeacher, sign_pm1

__all__ = ["Dimensions", "SeedHierarchy", "HiddenManifoldTeacher", "sign_pm1", "__version__"]
