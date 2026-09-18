import torch
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse import identity, diags 
from sklearn.neighbors import NearestNeighbors
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import sklearn
import anndata
import pandas as pd
import scanpy as sc
import sklearn.decomposition
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class spDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        x_rna, x_atac, 
        x_rna_smooth, x_atac_smooth, 
        spatial=None,     
        y=None, 
        batch_size=256
    ):
        super().__init__()
        self.x_rna = x_rna
        self.x_atac = x_atac
        self.x_rna_smooth = x_rna_smooth
        self.x_atac_smooth = x_atac_smooth
        self.spatial = spatial   
        self.y = y
        self.batch_size = batch_size

    def setup(self, stage=None):
        def to_float_tensor(x):
            return x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
    
        def to_long_tensor(x):
            return x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.long)
    
        # ✅ 全部在这里一次性处理好
        self.x_rna_t = to_float_tensor(self.x_rna)
        self.x_atac_t = to_float_tensor(self.x_atac)
        self.x_rna_smooth_t = to_float_tensor(self.x_rna_smooth)
        self.x_atac_smooth_t = to_float_tensor(self.x_atac_smooth)
    
        num_samples = self.x_rna_t.shape[0]
    
        # y
        if self.y is not None:
            self.y_t = to_long_tensor(self.y)   # 分类一般用 long
        else:
            self.y_t = torch.full((num_samples,), -1, dtype=torch.long)
    
        # spatial
        if self.spatial is not None:
            self.spatial_t = to_float_tensor(self.spatial)
        else:
            self.spatial_t = torch.zeros((num_samples, 2), dtype=torch.float32)
    
        self.indices_t = torch.arange(num_samples, dtype=torch.long)
    
        self.dataset = TensorDataset(
            self.x_rna_t,
            self.x_atac_t,
            self.x_rna_smooth_t,
            self.x_atac_smooth_t,
            self.spatial_t,
            self.y_t,
            self.indices_t
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

def make_sparse_tensor(adj):
    adj = adj.astype("float32").tocoo()
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adj.shape
    adj = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return adj


def precompute_SGC_pure(x, adj, n_layers, add_diag=True, mode="sgc"):


    deg = adj.sum(axis=1)
    
    if add_diag:
        deg = deg + 1  
        adj = adj + identity(n=x.shape[0])

    # D^-0.5
    deg = diags(deg.A1).power(-0.5)
    deg.data[deg.data == np.inf] = 0

    # A_norm = D^-0.5 * A_hat * D^-0.5
    adj = deg @ adj @ deg

    adj_tensor = make_sparse_tensor(adj)

    # --- (SGC Propagation) ---

    xs = x 
    
    sgc_list = []
    
    for i in range(n_layers):
        xs = torch.sparse.mm(adj_tensor, xs)
        sgc_list.append(xs)
    
    if mode == "sign":

        return torch.cat(sgc_list, dim=1)
    else:
        return sgc_list[-1]

def apply_sgc_pure(
    adata, 
    sample_key=None,  # e.g., "batch" or "sample"
    adj=None,
    k_neighbors=None, 
    radius=None, 
    n_layers=2, 
    spatial_key="spatial"
):
    print(f"--- SGC Smoothing ---")
    
    coords = adata.obsm[spatial_key]
    
    # 1. Build Adjacency Graph
    if adj is None:
        if sample_key is not None:
            print(f"Building spatial adjacency graph per sample: group column '{sample_key}'")
            samples = adata.obs[sample_key].unique()
            
            # Lists to collect edges from all local graphs
            row_idx, col_idx, data = [], [], []
            
            for sample in samples:
                # Get global row indices for the current slice in adata
                idx = np.where(adata.obs[sample_key] == sample)[0]
                sample_coords = coords[idx]
                
                if radius is not None:
                    nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(sample_coords)
                    local_adj = nbrs.radius_neighbors_graph(sample_coords, mode='connectivity')
                elif k_neighbors is not None:
                    # Defensive setting: truncate if cell count is less than k_neighbors
                    n_neigh = min(k_neighbors + 1, len(sample_coords))
                    nbrs = NearestNeighbors(n_neighbors=n_neigh, algorithm='ball_tree').fit(sample_coords)
                    local_adj = nbrs.kneighbors_graph(sample_coords, mode='connectivity')
                else:
                    raise ValueError("Either k_neighbors or radius must be specified")
                
                # Convert local graph indices to global graph indices
                local_adj = local_adj.tocoo()
                row_idx.extend(idx[local_adj.row])
                col_idx.extend(idx[local_adj.col])
                data.extend(local_adj.data)
                
            # Assemble a global sparse adjacency matrix using COO format, then convert to CSR
            adj = sp.coo_matrix((data, (row_idx, col_idx)), shape=(adata.n_obs, adata.n_obs)).tocsr()
            print(f"Multi-sample network construction completed, total nodes: {adata.n_obs}")

        else:
            # Original global construction logic (ignoring slices)
            if radius is not None:
                print(f"Using global radius graph: r={radius}")
                nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(coords)
                adj = nbrs.radius_neighbors_graph(coords, mode='connectivity')
            elif k_neighbors is not None:
                print(f"Using global kNN graph: k={k_neighbors}")
                nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(coords)
                adj = nbrs.kneighbors_graph(coords, mode='connectivity')
            else:
                raise ValueError("Either k_neighbors or radius must be specified")

    neighbor_counts = adj.sum(axis=1).A1
    print(f"Average number of neighbors: {neighbor_counts.mean():.2f}")
    
    # 2. Feature Tensor
    if sp.issparse(adata.X):
        x_data = adata.X.toarray().copy()
    else:
        x_data = adata.X.copy()
        
    x_tensor = torch.FloatTensor(x_data)

    # 3. SGC Smoothing
    smoothed_tensor = precompute_SGC_pure(x_tensor, adj, n_layers=n_layers)
    
    return smoothed_tensor.numpy(), adj


def TFIDF(count_mat): 
    """
    TF-IDF transformation for matrix.

    Parameters
    ----------
    count_mat
        numpy matrix with cells as rows and peak as columns, cell * peak.

    Returns
    ----------
    tfidf_mat
        matrix after TF-IDF transformation.

    divide_title
        matrix divided in TF-IDF transformation process, would be used in "inverse_TFIDF".

    multiply_title
        matrix multiplied in TF-IDF transformation process, would be used in "inverse_TFIDF".

    """
    count_mat = count_mat.T
    divide_title = np.tile(np.sum(count_mat,axis=0), (count_mat.shape[0],1))
    nfreqs = 1.0 * count_mat / divide_title
    multiply_title = np.tile(np.log(1 + 1.0 * count_mat.shape[1] / np.sum(count_mat,axis=1)).reshape(-1,1), (1,count_mat.shape[1]))
    tfidf_mat = scipy.sparse.csr_matrix(np.multiply(nfreqs, multiply_title)).T
    return tfidf_mat, divide_title, multiply_title


class tfidfTransformer:
    """
    TF-IDF Transformer for sparse count data.
    """
    def __init__(self):
        self.idf = None
        self.fitted = False

    def fit(self, X):
        """
        Compute IDF vector from input data.

        Parameters
        ----------
        X : array-like or sparse matrix
            Count matrix.
        """
        self.idf = X.shape[0] / (1e-8+X.sum(axis=0))
        self.fitted = True

    def transform(self, X):
        """
        Apply TF-IDF transformation using precomputed IDF.

        Parameters
        ----------
        X : array-like or sparse matrix
            Count matrix to transform.

        Returns
        -------
        Transformed matrix.
        """
        if not self.fitted:
            raise RuntimeError("Transformer was not fitted on any data")
        if sp.issparse(X):
            tf = X.multiply(1 / (1e-8+X.sum(axis=1)))
            return tf.multiply(self.idf)
        else:
            tf = X / (1e-8+X.sum(axis=1, keepdims=True))
            return tf * self.idf

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like or sparse matrix

        Returns
        -------
        Transformed matrix.
        """
        self.fit(X)
        return self.transform(X)



def sparse_log1p_scale(X, scale=1e4):
    """
    Apply log1p transformation to sparse or dense matrix, scaled by a factor.

    Parameters
    ----------
    X : Union[scipy.sparse.spmatrix, np.ndarray]
        Input expression matrix.
    scale : float, default=1e4
        Scaling factor applied before log1p.

    Returns
    -------
    Transformed matrix (same type as input)
    """
    if scipy.sparse.issparse(X):
        X = X.copy()
        X.data = np.log1p(X.data * scale)
        return X
    else:
        return np.log1p(X * scale)


# optional, other reasonable preprocessing steps also ok
class lsiTransformer:
    """
    Latent Semantic Indexing (LSI) pipeline for dimensionality reduction.

    Parameters
    ----------
    n_components : int
        Number of SVD components.
    drop_first : bool
        Whether to drop the first principal component.
    use_highly_variable : bool or None
        Whether to subset to highly variable features.
    log : bool
        Whether to apply log1p transformation.
    norm : bool
        Whether to normalize features.
    z_score : bool
        Whether to z-score features.
    tfidf : bool
        Whether to apply TF-IDF normalization.
    svd : bool
        Whether to apply SVD transformation.
    use_counts : bool
        Use `.layers['counts']` instead of `.X` for data.
    pcaAlgo : str
        SVD backend.
    """

    def __init__(
        self, n_components: int = 20, drop_first=True, use_highly_variable=None, log=True, norm=True, z_score=True,
        tfidf=True, svd=True, use_counts=False, pcaAlgo='arpack'
    ):  

        self.drop_first = drop_first
        self.n_components = n_components + drop_first
        self.use_highly_variable = use_highly_variable

        self.log = log
        self.norm = norm
        self.z_score = z_score
        self.svd = svd
        self.tfidf = tfidf
        self.use_counts = use_counts

        self.tfidfTransformer = tfidfTransformer()
        self.normalizer = sklearn.preprocessing.Normalizer(norm="l1")
        self.pcaTransformer = sklearn.decomposition.TruncatedSVD(
            n_components=self.n_components, random_state=777, algorithm=pcaAlgo
        )
        self.fitted = None

    def fit(self, adata: anndata.AnnData):
        """
        Fit the transformer on AnnData object.
        """
        if self.use_highly_variable is None:
            self.use_highly_variable = "highly_variable" in adata.var
        adata_use = (
            adata[:, adata.var["highly_variable"]]
            if self.use_highly_variable
            else adata
        )
        if self.use_counts:
            X = adata_use.layers['counts']
        else:
            X = adata_use.X
        if self.tfidf:
            X = self.tfidfTransformer.fit_transform(X)
        # if scipy.sparse.issparse(X):
        #     X = X.A.astype("float32")
        if self.norm:
            X = self.normalizer.fit_transform(X)
        if self.log:
            # X = np.log1p(X * 1e4)    # L1-norm and target_sum=1e4 and log1p
            X = sparse_log1p_scale(X, 1e4)
        self.pcaTransformer.fit(X)
        self.fitted = True

    def transform(self, adata):
        """
        Transform AnnData using fitted transformer.
        """
        if not self.fitted:
            raise RuntimeError("Transformer was not fitted on any data")
        adata_use = (
            adata[:, adata.var["highly_variable"]]
            if self.use_highly_variable
            else adata
        )
        if self.use_counts:
            X_pp = adata_use.layers['counts']
        else:
            X_pp = adata_use.X
        if self.tfidf:
            X_pp = self.tfidfTransformer.transform(X_pp)
        # if scipy.sparse.issparse(X_pp):
        #     X_pp = X_pp.A.astype("float32")
        if self.norm:
            X_pp = self.normalizer.transform(X_pp)
        if self.log:
            # X_pp = np.log1p(X_pp * 1e4)
            X_pp = sparse_log1p_scale(X_pp, 1e4)
        if self.svd:
            X_pp = self.pcaTransformer.transform(X_pp)
        if self.z_score:
            X_pp -= X_pp.mean(axis=1, keepdims=True)
            X_pp /= (1e-8+X_pp.std(axis=1, ddof=1, keepdims=True))
        pp_df = pd.DataFrame(X_pp, index=adata_use.obs_names).iloc[
            :, int(self.drop_first) :
        ]
        return pp_df

    def fit_transform(self, adata):
        """
        Fit and transform AnnData.
        """
        self.fit(adata)
        return self.transform(adata)
   
# CLR-normalization     
def clr_normalize(adata):
    """
    Perform centered log-ratio (CLR) normalization on count data.

    Parameters
    ----------
    adata : AnnData
        Input data with count matrix in `.X`.

    Returns
    -------
    adata : AnnData
        Normalized AnnData object.
    """
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if sp.issparse(adata.X) else np.array(adata.X))
    )
    # sc.pp.pca(adata, n_comps=min(50, adata.n_vars-1))
    return adata




def process_adata_for_training(adata, adata1, label_key="cell_type"):
    if not all(adata.obs_names == adata1.obs_names):
        print("Aligning adata1 to adata based on obs_names...")
        adata1 = adata1[adata.obs_names, :].copy()

    if sp.issparse(adata.X):
        x_rna = adata.X.toarray()
    else:
        x_rna = adata.X

    if sp.issparse(adata1.X):
        x_atac = adata1.X.toarray()
    else:
        x_atac = adata1.X
    print(f" - RNA shape: {x_rna.shape}")
    print(f" - ATAC shape: {x_atac.shape}")

    if label_key is not None:
        raw_labels = adata.obs[label_key].values
        le = LabelEncoder()
        y = le.fit_transform(raw_labels)
        n_classes = len(le.classes_)
        print(f" - Number of classes: {n_classes}")
        return x_rna, x_atac, y, n_classes
    else:
        return x_rna, x_atac
