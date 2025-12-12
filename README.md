# SCMI-Net

This repository contains the code and utilities for SCMI-Net: Semantic Constraints and Modal Interaction Network for Multimodal Emotion Recognition. The code base includes model definitions, datasets, training scripts, evaluation and visualization utilities used in the associated paper.

## Structure 
- `model/` - PyTorch model definitions (SCMI, ablation variants, losses).
- `src/` - training/evaluation runners and plotting utilities.
- `utils/` - dataset loaders, preprocessing and helper utilities.
- `pre/` - utilities for extracting precomputed embeddings (e.g., roberta).
- `pretained_model/` - expected local pretrained models (e.g., `roberta-base`, `wav2vec2-base`).
- `results/` - example results / saved outputs.
- `test/` - small test scripts.

## Requirements
See `requirements.txt` for Python package requirements. Typical setup:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Pretrained models and data
- Put local pretrained models under pretained_model/:
pretained_model/roberta-base/
pretained_model/wav2vec2-base/ (or other wav2vec models)
- Datasets (IEMOCAP / MELD / SAVEE) should be organized in paths referenced by the CLI arguments in the scripts under src/ and pre/.