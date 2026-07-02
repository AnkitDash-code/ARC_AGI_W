# Kaggle Upload Manifest

Use this project as two Kaggle inputs:

1. The official competition dataset, already mounted by Kaggle:
   `/kaggle/input/competitions/arc-prize-2026-arc-agi-2`
2. A custom Kaggle Dataset containing this Project Mythos code.

## Required Code Files

Upload these files/directories as the Project Mythos code dataset:

```text
project_mythos_kaggle_pipeline.ipynb
pyproject.toml
src/
configs/
scripts/
README.md
```

The minimum required runtime directory is:

```text
src/mythos/
```

The notebook now searches these locations automatically:

```text
./src
/kaggle/working/src
/kaggle/working/ARC-AGI-2/src
/kaggle/input/*/src
/kaggle/input/*/ARC-AGI-2/src
```

## Optional Local Test Files

Upload these only if you want to run the toy fixture tests inside Kaggle:

```text
data/toy/
tests/
```

They are not needed for generating the competition `submission.json`.

## Not Required For Kaggle

These are documentation/local planning files and do not need to be uploaded for a submission run:

```text
project_mythos_master_plan.html
runs/
.pytest_cache/
```

## Model Inputs

For `MODEL_MODE = "fallback"`, no model files are required; missing models are reported and the pipeline uses fallback adapters.

For `MODEL_MODE = "strict"`, upload separate Kaggle Datasets for every checkpoint/repo and set these env vars in the notebook:

```python
os.environ["IJEPA_REPO_DIR"] = "/kaggle/input/<ijepa-code-dataset>/ijepa"
os.environ["IJEPA_CHECKPOINT_PATH"] = "/kaggle/input/<jepa-checkpoint-dataset>/checkpoint.pt"
os.environ["HRM_TEXT_REPO_DIR"] = "/kaggle/input/<hrm-text-code-dataset>/hrm-text"
os.environ["HRM_TEXT_CHECKPOINT_PATH"] = "/kaggle/input/<hrm-text-checkpoint-dataset>/checkpoint.pt"
os.environ["WORLD_MODEL_CHECKPOINT_PATH"] = "/kaggle/input/<world-model-dataset>/world_model.pt"
os.environ["TTT_LORA_CHECKPOINT_PATH"] = "/kaggle/input/<lora-dataset>/lora.pt"
os.environ["HRM_REPO_DIR"] = "/kaggle/input/<hrm-code-dataset>/HRM"
os.environ["HRM_CHECKPOINT_PATH"] = "/kaggle/input/<hrm-checkpoint-dataset>/checkpoint.pt"
```

Current limitation: checkpoints are loaded and attached to planned stages, but full real-model forward/inference calls are not implemented yet.
