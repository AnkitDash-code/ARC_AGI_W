from __future__ import annotations

import pytest

from mythos.solvers.hrm import HRMEnvironment, HRMEnvironmentError


def test_hrm_environment_requires_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HRM_REPO_DIR", raising=False)
    monkeypatch.delenv("HRM_CHECKPOINT_PATH", raising=False)

    with pytest.raises(HRMEnvironmentError, match="HRM_REPO_DIR"):
        HRMEnvironment.from_env()
