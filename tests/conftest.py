from pathlib import Path

import pytest
import torch

from resemreason.pipeline import ReSemReasonPipeline


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def pipeline(repo_root: Path) -> ReSemReasonPipeline:
    return ReSemReasonPipeline.from_yaml(repo_root / "configs" / "default.yaml")


# Tiny numerical tests do not benefit from a machine-wide thread pool.

torch.set_num_threads(1)
