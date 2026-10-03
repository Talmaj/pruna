from pruna import PrunaModel
from pruna.algorithms.torchao import Torchao

from .base_tester import AlgorithmTesterBase
from .utils import configure_tiny_flux2_inference


class TestTorchao(AlgorithmTesterBase):
    """Test the torchao quantizer."""

    models = ["flux_tiny_random", "flux2_tiny_random", "sd_tiny_random"]
    reject_models = ["dummy_lambda"]
    allow_pickle_files = False
    algorithm_class = Torchao
    metrics = ["cmmd"]

    def post_load_hook(self, model: PrunaModel) -> None:
        """Keep tiny Flux.2 generation inside CPU test limits."""
        configure_tiny_flux2_inference(model)
