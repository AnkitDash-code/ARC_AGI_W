# Kaggle Upload Manifest

For the current standalone delivery path, upload one notebook:

```text
project_mythos_kaggle_pipeline_standalone.ipynb
```

The official competition dataset is mounted by Kaggle at:

```text
/kaggle/input/competitions/arc-prize-2026-arc-agi-2
```

The standalone notebook embeds `src/mythos` and writes it into
`/kaggle/working/project_mythos_embedded/src` at runtime.

## Required Code Files

Only required if you choose not to use the standalone notebook:

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

## Model Inputs And Training Outputs

For `MODEL_MODE = "fallback"`, no model files are required; missing models are reported and the pipeline uses fallback adapters.

For `MODEL_MODE = "strict"`, upload separate Kaggle Datasets for every checkpoint/repo and set these env vars in the notebook:

```python
os.environ["IJEPA_CHECKPOINT_PATH"] = "/kaggle/input/<ijepa-hf-snapshot>/model.safetensors"
os.environ["IJEPA_PROJECTION_CHECKPOINT_PATH"] = "/kaggle/input/<projection-dataset>/ijepa_projection.pt"
os.environ["HRM_TEXT_REPO_DIR"] = "/kaggle/input/<hrm-text-code-dataset>/hrm-text"
os.environ["HRM_TEXT_CHECKPOINT_PATH"] = "/kaggle/input/<hrm-text-checkpoint-dataset>/checkpoint.pt"
os.environ["WORLD_MODEL_CHECKPOINT_PATH"] = "/kaggle/input/<world-model-dataset>/world_model.pt"
os.environ["TTT_LORA_CHECKPOINT_PATH"] = "/kaggle/input/<lora-dataset>/lora.pt"
os.environ["HRM_REPO_DIR"] = "/kaggle/input/<hrm-code-dataset>/HRM"
os.environ["HRM_CHECKPOINT_PATH"] = "/kaggle/input/<hrm-checkpoint-dataset>/checkpoint.pt"
```

The standalone notebook can also train these local artifacts under
`/kaggle/working/mythos_checkpoints` when `RUN_TRAINING_STAGES = True`:

```text
ijepa_projection.pt
world_model.pt
ttt_lora_smoke.pt
```

Real I-JEPA execution is guarded by `ENABLE_REAL_JEPA = True`,
`IJEPA_CHECKPOINT_PATH`, `transformers`, `torch`, and `Pillow`. It uses the
transformers-native `facebook/ijepa_vith14_1k` snapshot and does not require the
archived `facebookresearch/ijepa` repo.

Real HRM execution is guarded by `ENABLE_REAL_HRM_INFERENCE = True`,
`HRM_REPO_DIR`, `HRM_CHECKPOINT_PATH`, `all_config.yaml` next to the checkpoint,
HRM dependencies, and CUDA. HRM attention depends on `flash-attn`; pre-stage a
wheel built against Kaggle's PyTorch/CUDA image instead of building it during
the final rerun. Default fallback mode keeps producing a schema-valid
`submission.json` when that path is not available.

Internet downloads are disabled by default because Kaggle reruns are
internet-disabled. To use the public defaults during an interactive notebook
session with internet enabled, manually set:

```python
AUTO_DOWNLOAD_GIT_CODE = True
AUTO_DOWNLOAD_HF_MODELS = True
AUTO_DOWNLOAD_DIRECT_CHECKPOINTS = True
```

## Verified Public Defaults

The standalone notebook includes these verified defaults:

```python
os.environ.setdefault("HRM_GIT_REPO_URL", "https://github.com/sapientinc/HRM.git")
os.environ.setdefault("HRM_HF_REPO_ID", "sapientinc/HRM-checkpoint-ARC-2")
os.environ.setdefault("HRM_HF_CHECKPOINT_GLOB", "checkpoint")
os.environ.setdefault("IJEPA_HF_REPO_ID", "facebook/ijepa_vith14_1k")
os.environ.setdefault("IJEPA_HF_CHECKPOINT_GLOB", "model.safetensors")
os.environ.setdefault("HRM_TEXT_HF_REPO_ID", "sapientinc/HRM-Text-1B")
os.environ.setdefault("HRM_TEXT_HF_CHECKPOINT_GLOB", "model.safetensors")
```

The following are intentionally unset and should be trained or supplied later:

```text
IJEPA_PROJECTION_HF_REPO_ID
WORLD_MODEL_HF_REPO_ID
TTT_LORA_HF_REPO_ID
```
