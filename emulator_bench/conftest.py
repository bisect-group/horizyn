from pathlib import Path

import pytest

from emulator_bench import cache_features
from emulator_bench.dataset_adapter import SplitDir

FIXTURE_SPLIT = Path(__file__).parent / "tests" / "fixtures" / "split"


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device (run with verify_equivalence.sh --gpu)")


@pytest.fixture(scope="session")
def fixture_data_dir(tmp_path_factory) -> Path:
    """Wrapper data files of the synthetic fixture split. Protein vectors are deterministic stand-ins for ProtT5."""
    out_root = tmp_path_factory.mktemp("fixture_out")
    cache_features.prepare_split(
        SplitDir("fixture", "all", FIXTURE_SPLIT),
        out_root=out_root,
        embedding_source="deterministic",
        num_workers=1,
    )
    return out_root / "enzyme_retrieval" / "fixture" / "all" / "data"
