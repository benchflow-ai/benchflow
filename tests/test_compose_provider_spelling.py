"""A provider service's refusal names its program in either Compose spelling.

Guards the provider refusal message from d6076e88 (BenchFlow 0.9.1), which
printed "runs the program None" under Compose 2.38:
``docker compose config --format json`` from Compose 2.38.2 (GitHub's
ubuntu-24.04 runner) prints a provider's program as ``"driver"``; its YAML
output, and Compose 2.40.3, print ``"type"``. The gate test in
``test_compose_oracle`` reads whichever Compose the machine has.
"""

from __future__ import annotations

import pytest

from benchflow.task.compose_env import _host_settings


@pytest.mark.parametrize("key", ["type", "driver"])
def test_a_provider_names_its_program_in_either_spelling(key):
    document = {
        "services": {
            "main": {
                "image": "alpine",
                "provider": {key: "./hook.sh", "options": {"x": ["1"]}},
            }
        }
    }
    assert list(_host_settings(document)) == [
        "runs the program ./hook.sh on this machine for service main (provider)"
    ]
