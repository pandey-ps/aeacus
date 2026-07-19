```bash
pip install aeacus
```

from source:

```bash
git clone https://github.com/pandey-ps/aeacus.git
cd aeacus
pip install -e .
```

### Usage

```python
from aeacus import Profiler

result = Profiler(test_input="query.h5ad").load().profile()
result.obs["malignancy_call"].value_counts()
```

### Input

| Format | Notes |
|---|---|
| `.h5ad` | genes in `var_names`, cells in `obs_names` |
| `.txt` / `.tsv` | rows = genes, columns = cells |
| `.csv` | rows = genes, columns = cells |
| `AnnData` object | passed directly |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `test_input` | required | path to data or `AnnData` object |
| `pretrain_dir` | auto | path to folder with `moe.pt`, `geneorder.tsv`, `train_mean.npy`, `train_std.npy`, `config.json`.|
| `norm_type` | `False` | normalization to apply before inference (see below) |
| `use_raw` | `False` | if `True`, reads from `adata.raw.X` instead of `adata.X` |
| `batch_size` | `8192` | cells per batch, lower if out of memory |
| `device` | auto | `"cuda"` or `"cpu"` |

### `norm_type` 

model was trained on **CPM + log1p** normalized data, choose `norm_type` based on your input:

| Input | `norm_type` |
|---|---|
| raw `UMI` counts (10x, etc.) | `"cpm_log1p"` (or `True`) |
| `CPM + log1p` normalized | `False` (default) or `"already_normalized"` |
| `TPM` data (smart-seq, etc.) | `"tpm_log1p"` |

**Examples:**

```python
# raw counts - normalize 
Profiler(test_input="raw_counts.h5ad", norm_type="cpm_log1p")

# normalized - skip
Profiler(test_input="normalized.h5ad", norm_type=False)

# TPM - log1p
Profiler(test_input="tpm_data.h5ad", norm_type="tpm_log1p")
```


### `use_raw`

`use_raw` chooses which data slot to read from:

| Input | `use_raw` | `norm_type` |
|---|---|---|
| raw counts in `.X`, no `.raw` | `False` | `"cpm_log1p"` |
| raw counts in `.raw`, normalized in `.X` | `True` | `"cpm_log1p"` |
| normalized in `.X` | `False` | `False` |

**Example - AnnData with raw counts stored in `.raw`:**

```python
Profiler(
    test_input="adata.h5ad",
    use_raw=True,          # read from adata.raw.X
    norm_type="cpm_log1p", # then normalize
)
```

### Inference

```python
result = Profiler(test_input="data.h5ad", norm_type="cpm_log1p").load().profile()
```

### Output

all predictions are added to `result.obs`:

| Column | Description |
|---|---|
| `malignancy_call` | `"Malignant"` or `"Normal"` |
| `malignancy_score` | probability per cell |
| `normal_expert_weight` | weight assigned to the normal expert |
| `malignant_expert_weight` | weight assigned to the malignant expert |gating |

```python
result.obs[["malignancy_call", "malignancy_score"]].head()
```

### Note

- **missing genes** are filled with zeros after aligning to `geneorder.tsv`; warning showed if >20% of model genes are missing.

