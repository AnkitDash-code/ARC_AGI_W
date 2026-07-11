from __future__ import annotations

from pathlib import Path

from mythos.arc import load_challenges
from mythos.features import grid_to_feature_vector, max_pairwise_cosine
from mythos.solvers.pipeline import PlannedPipelineSolver
from mythos.training import (
    JepaProjectionConfig,
    WorldModelConfig,
    adaptive_ttt_should_stop,
    load_projection_checkpoint,
    run_ttt_lora_smoke,
    train_jepa_projection,
    train_world_model,
)


ROOT = Path(__file__).resolve().parents[1]


def test_jepa_projection_checkpoint_trains_and_projects(tmp_path: Path) -> None:
    tasks = load_challenges(ROOT / "data" / "toy" / "challenges.json")
    checkpoint = tmp_path / "ijepa_projection.pt"

    result = train_jepa_projection(
        tasks.values(),
        checkpoint_path=checkpoint,
        config=JepaProjectionConfig(input_dim=32, output_dim=16),
        steps=10,
        device="cpu",
    )

    model = load_projection_checkpoint(checkpoint, device="cpu")
    assert checkpoint.exists()
    assert result.examples > 0
    assert result.final_loss <= result.initial_loss
    assert model.proj.out_features == 16


def test_grid_embedding_diversity_smoke() -> None:
    tasks = load_challenges(ROOT / "data" / "toy" / "challenges.json")
    vectors = [
        grid_to_feature_vector(example.input, dim=128)
        for task in tasks.values()
        for example in task.train[:1]
    ]

    assert max_pairwise_cosine(vectors) < 0.99


def test_world_model_checkpoint_trains(tmp_path: Path) -> None:
    tasks = load_challenges(ROOT / "data" / "toy" / "challenges.json")
    checkpoint = tmp_path / "world_model.pt"

    result = train_world_model(
        tasks.values(),
        checkpoint_path=checkpoint,
        config=WorldModelConfig(z_dim=16, rule_dim=4, hidden_dim=32),
        steps=20,
        device="cpu",
    )

    assert checkpoint.exists()
    assert result.examples > 0
    assert result.final_loss <= result.initial_loss


def test_lora_ttt_smoke_updates_only_lora_params(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ttt_lora.pt"

    result = run_ttt_lora_smoke(
        rank=2,
        steps=15,
        dim=8,
        batch_size=4,
        device="cpu",
        checkpoint_path=checkpoint,
    )

    assert checkpoint.exists()
    assert result.backbone_frozen
    assert result.injected_modules
    assert result.first_backward_seconds >= 0.0


def test_adaptive_ttt_stop_detects_flat_loss() -> None:
    assert adaptive_ttt_should_stop([1.0, 0.9, 0.89999, 0.89998, 0.89997], min_delta=1e-3, patience=3)
    assert not adaptive_ttt_should_stop([1.0, 0.9, 0.8], min_delta=1e-3, patience=3)


def test_pipeline_uses_projection_and_world_checkpoints(tmp_path: Path, monkeypatch) -> None:
    tasks = load_challenges(ROOT / "data" / "toy" / "challenges.json")
    projection = tmp_path / "ijepa_projection.pt"
    world = tmp_path / "world_model.pt"
    train_jepa_projection(
        tasks.values(),
        checkpoint_path=projection,
        config=JepaProjectionConfig(input_dim=32, output_dim=16),
        steps=2,
        device="cpu",
    )
    train_world_model(
        tasks.values(),
        checkpoint_path=world,
        config=WorldModelConfig(z_dim=16, rule_dim=4, hidden_dim=32),
        steps=2,
        device="cpu",
    )
    monkeypatch.setenv("IJEPA_PROJECTION_CHECKPOINT_PATH", str(projection))
    monkeypatch.setenv("WORLD_MODEL_CHECKPOINT_PATH", str(world))

    solver = PlannedPipelineSolver()
    prediction = solver.solve(tasks["toy_identity"])

    assert prediction.task_id == "toy_identity"
    assert solver.last_trace is not None
    statuses = {stage.name: stage.status for stage in solver.last_trace.stages}
    assert statuses["encode_jepa"] == "projection_loaded"
    assert statuses["simulate_world_model"] == "model_loaded"
