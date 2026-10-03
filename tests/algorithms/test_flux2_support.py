"""Model-check coverage for Flux.2 pipelines.

Full smash, save and reload of the algorithms that run on Flux.2 is covered by
the algorithm testers (``flux2_tiny_random``). This module checks the split
between Flux.1 and Flux.2 and the algorithms that must reject Flux.2.
"""

import diffusers
import pytest

from pruna.algorithms.fastercache import FasterCache
from pruna.algorithms.fora import FORA
from pruna.algorithms.hqq_diffusers import HQQDiffusers
from pruna.algorithms.hyper import Hyper
from pruna.algorithms.pab import PAB
from pruna.algorithms.qkv_diffusers import QKVFusing
from pruna.algorithms.static_fp8_diffusers import StaticFp8Diffusers
from pruna.algorithms.time_aware_fp8_diffusers import TimeAwareFp8Diffusers
from pruna.algorithms.torchao import Torchao
from pruna.engine.model_checks import is_flux2_pipeline, is_flux_pipeline

pytestmark = pytest.mark.cpu


def test_flux_pipeline_checks_split_generations() -> None:
    """Flux.1 and Flux.2 pipeline classes must not share a model check."""
    flux1 = object.__new__(diffusers.FluxPipeline)
    assert is_flux_pipeline(flux1)
    assert not is_flux2_pipeline(flux1)

    if not hasattr(diffusers, "Flux2Pipeline"):
        pytest.skip("installed diffusers has no Flux.2 pipelines")

    flux2 = object.__new__(diffusers.Flux2Pipeline)
    assert is_flux2_pipeline(flux2)
    assert not is_flux_pipeline(flux2)

    if hasattr(diffusers, "Flux2KleinPipeline"):
        klein = object.__new__(diffusers.Flux2KleinPipeline)
        assert is_flux2_pipeline(klein)
        assert not is_flux_pipeline(klein)


@pytest.mark.parametrize("model_fixture", ["flux2_tiny_random"], indirect=True)
def test_flux2_algorithm_model_checks(model_fixture: tuple) -> None:
    """Quantizers and block cachers accept Flux.2; Flux.1-specific algorithms do not."""
    model, _ = model_fixture

    assert is_flux2_pipeline(model)
    assert not is_flux_pipeline(model)

    assert FORA().model_check_fn(model)
    assert PAB().model_check_fn(model)
    assert FasterCache().model_check_fn(model)
    assert Torchao().model_check_fn(model)
    assert HQQDiffusers().model_check_fn(model)
    assert StaticFp8Diffusers().model_check_fn(model)
    assert TimeAwareFp8Diffusers().model_check_fn(model)

    # Hyper-SD LoRAs and QKV fusion target Flux.1 modules.
    assert not Hyper().model_check_fn(model)
    assert not QKVFusing().model_check_fn(model)
