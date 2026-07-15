from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp #in case we get sparse expression matrices


def load_features(path): # gene order to list
    path = Path(path) 
    features = pd.read_csv(path, header=None, sep="\t").iloc[:, 0].astype(str).tolist()
    if features and features[0].strip().lower() in {"gene", "genes"}:
        features = features[1:]
    return features

# save as sparse CSR
def load_input(input_data, use_raw=False, norm_type=False):
    if isinstance(input_data, sc.AnnData):
        adata = input_data.copy()
    else:
        input_path = str(input_data)
        if input_path.endswith(".h5ad"):
            adata = sc.read_h5ad(input_path)
        elif input_path.endswith((".txt", ".txt.gz", ".tsv", ".tsv.gz")):
            adata = sc.AnnData(pd.read_csv(input_path, sep="\t", index_col=0).T)
            adata.X = sp.csr_matrix(adata.X)
        elif input_path.endswith((".csv", ".csv.gz")):
            adata = sc.AnnData(pd.read_csv(input_path, sep=",", index_col=0).T)
            adata.X = sp.csr_matrix(adata.X)
        else:
            raise ValueError("Input must be an AnnData object, .h5ad, .txt/.tsv, or .csv file.")
    # unique to prevent duplicate 
    adata.var_names_make_unique()
    # if user requested for adata.raw.X
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True, but AnnData.raw is not available.")
        adata.X = adata.raw.X.copy()

    # normalize the input (CMPN/log1p or TPM/log1p) as in user arg.
    normalize_expression(adata, norm_type)

    return adata

# return if already normalized
# check docs for profiler
def normalize_expression(adata, norm_type=False):
    if norm_type is True:
        norm_type = "cpm_log1p"

    if norm_type in (False, None):
        return adata

    norm_type = str(norm_type).strip().lower()
    if norm_type in {"false", "none", "already_normalized"}:
        return adata

    if norm_type in {"cpm_log1p", "umi", "umi_cpm_log1p"}:
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)
    elif norm_type in {"tpm_log1p", "tpm"}:
        sc.pp.log1p(adata)
    else:
        raise ValueError(
            "norm_type must be one of False, 'false', 'already_normalized', "
            "'cpm_log1p', or 'tpm_log1p'."
        )

    return adata

# to match the input gene to model's gene order
# also return missing genes
def feature_indexer(var_names, ordered_features):
    var_to_idx = {str(gene): i for i, gene in enumerate(var_names)}
    target_idx = []
    source_idx = []
    missing = []

    for i, gene in enumerate(ordered_features):
        source = var_to_idx.get(str(gene))
        if source is None:
            missing.append(str(gene))
        else:
            target_idx.append(i)
            source_idx.append(source)

    return np.asarray(target_idx), np.asarray(source_idx), missing

# create batches for the model 
# sparse matrix to dense numpy array
# batch to numpy array with gene order
def dense_batch_aligned(adata, batch_indices, target_idx, source_idx, n_features):
    x = np.zeros((len(batch_indices), n_features), dtype=np.float32)
    if len(source_idx) == 0:
        return x

    batch = adata.X[batch_indices][:, source_idx]
    if sp.issparse(batch):
        batch = batch.toarray()
    else:
        batch = np.asarray(batch)

    x[:, target_idx] = batch.astype(np.float32, copy=False)
    return x

# training set norm
def zscore_batch(x, train_mean, train_std):
    return (x - train_mean) / train_std
