from pruna import PrunaModel
from pruna.algorithms.fora import FORA

from .base_tester import AlgorithmTesterBase
from .utils import configure_tiny_flux2_inference


class TestFORA(AlgorithmTesterBase):
    """Test the fora algorithm."""

    models = ["flux_tiny_random", "flux2_tiny_random"]
    reject_models = ["opt_tiny_random"]
    allow_pickle_files = False
    algorithm_class = FORA
    metrics = ["lpips", "throughput"]

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert hasattr(model, "cache_helper")

    def post_load_hook(self, model: PrunaModel) -> None:
        """Keep tiny Flux.2 generation inside CPU test limits."""
        configure_tiny_flux2_inference(model)
