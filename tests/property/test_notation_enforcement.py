"""Contract enforcement: bare `alpha`/`gamma`/`L` are rejected at the config
boundary and absent from the active package's public interfaces
(docs/NOTATION.md)."""

import re
from pathlib import Path

import pytest

from maskeddiffusion.config import load_config

SRC = Path(__file__).parents[2] / "src" / "maskeddiffusion"


def write_toml(tmp_path, body: str):
    p = tmp_path / "config.toml"
    p.write_text(body)
    return p


BASE = """
[seeds]
base_seed = 1
"""


@pytest.mark.parametrize("key", ["alpha", "gamma", "L"])
def test_forbidden_keys_rejected(tmp_path, key):
    cfg = write_toml(
        tmp_path,
        f"""
[dimensions]
latent_dim = 4
aspect_ratio = 2.0
sample_ratio = 2.0
{key} = 1.0
"""
        + BASE,
    )
    with pytest.raises(ValueError, match="forbidden"):
        load_config(cfg)


def test_forbidden_keys_rejected_in_nested_tables(tmp_path):
    cfg = write_toml(
        tmp_path,
        """
[dimensions]
latent_dim = 4
aspect_ratio = 2.0
sample_ratio = 2.0

[training]
alpha = 0.5
"""
        + BASE,
    )
    with pytest.raises(ValueError, match="forbidden"):
        load_config(cfg)


def test_contract_names_accepted(tmp_path):
    cfg = write_toml(
        tmp_path,
        """
[dimensions]
latent_dim = 4
aspect_ratio = 2.0
sample_ratio = 2.0
"""
        + BASE,
    )
    run = load_config(cfg)
    assert run.dimensions.visible_dim == 8
    assert run.dimensions.train_size == 8


def test_no_bare_alpha_identifiers_in_active_package():
    """No function parameter or variable named exactly `alpha`/`gamma` in the
    new package (mathematical docs/comments are exempt; identifiers are not)."""
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        # strip comments and docstrings crudely: drop lines starting with #
        code_lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        for ln in code_lines:
            if re.search(r"\b(alpha|gamma)\b\s*[=:,)]", ln) and '"' not in ln and "'" not in ln:
                offenders.append(f"{path.name}: {ln.strip()}")
    assert offenders == [], f"bare alpha/gamma identifiers found: {offenders}"
