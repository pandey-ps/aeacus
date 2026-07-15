The `scMoE/` folder contains inference code. The `model/` folder contains the trained weights and normalization artifacts.

## Install

From the repository root:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Input Format

Input can be:

- `.h5ad`
- `.txt` / `.tsv`
- `.csv`
- an `AnnData` object

For text files, rows should be genes and columns should be cells. For `.h5ad`, genes should be in `adata.var_names` and cells in `adata.obs_names`.

## Normalization

To use the same expression scale expected by the model:

- Already normalized: `norm_type="False"`
- Raw UMI counts: `norm_type="cpm_log1p"`
- TPM values: `norm_type="tpm_log1p"`

Do not z-score on new data. The model internally applies the training-set mean and standard deviation


## Inference

For raw UMI count data:

```python
from scMoE import Profiler

profiler = Profiler(
    test_input="query.h5ad",
    pretrain_dir="model",
    norm_type="cpm_log1p",
)

result_adata = profiler.load().profile()
```

For TPM data:

```python
from scMoE import Profiler

result_adata = (
    Profiler(
        test_input="query.h5ad",
        pretrain_dir="model",
        norm_type="tpm_log1p",
    )
    .load()
    .profile()
)
```

For data already normalized:

```python
from scMoE import Profiler

result_adata = (
    Profiler(
        test_input="query.h5ad",
        pretrain_dir="model",
        norm_type="False",
    )
    .load()
    .profile()
)
```

## Output

Predictions are added to `result_adata.obs`:

```text
malignancy_call      Normal or Malignant
malignancy_score     malignancy probability-like score
primary_expert       expert with the highest gate weight
primary_expert_label Normal or Malignant label for primary_expert
gate_entropy         uncertainty/spread of gate weights
normal_expert_weight gate weight assigned to the normal expert
malignant_expert_weight gate weight assigned to the malignant expert
expert_weight_0      same value as normal_expert_weight
expert_weight_1      same value as malignant_expert_weight
normal_expert_logit raw logit from the normal expert
malignant_expert_logit raw logit from the malignant expert
```

View results:

```python
result_adata.obs["malignancy_call"].head()
```

View all other scMoE output columns:

```python
prediction_cols = [
    "malignancy_call",
    "malignancy_score",
    "primary_expert",
    "primary_expert_label",
    "gate_entropy",
    "normal_expert_weight",
    "malignant_expert_weight",
    "expert_weight_0",
    "expert_weight_1",
    "normal_expert_logit",
    "malignant_expert_logit",
]

result_adata.obs[prediction_cols].head()
```

## Note

- `geneorder.tsv` controls the model gene order. Missing genes are filled with zero after alignment.
