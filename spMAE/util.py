import scanpy as sc
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

def get_modality_plot_data(emb, alpha1, alpha2, labels, top_n_dims=30):
    """
    Analyzes modality importance by assigning embedding dimensions to specific cell types
    based on differential expression (T-test) and calculating associated attention weights.

    Parameters:
    - emb: Combined embedding matrix (N_cells, D_dims)
    - alpha1: Attention weights for modality 1 (N_cells, D_dims)
    - alpha2: Attention weights for modality 2 (N_cells, D_dims)
    - labels: Cell type labels for each cell (N_cells,)
    - top_n_dims: Maximum number of core dimensions to consider per cell type

    Returns:
    - pd.DataFrame: Long-format dataframe ready for seaborn violin plots
    """
    
    # Ensure all inputs are numpy arrays for consistent indexing
    emb = np.array(emb)
    alpha1 = np.array(alpha1)
    alpha2 = np.array(alpha2)
    labels = np.array(labels)
    
    unique_types = np.unique(labels)
    num_dims = emb.shape[1]
    
    # Step 1: Dimension Assignment
    # Identify which cell type "owns" each dimension based on the highest positive T-statistic
    dim_records = []
    for d in range(num_dims):
        best_ct = None
        max_t = 0
        
        for ct in unique_types:
            mask_target = (labels == ct)
            mask_others = ~mask_target
            
            # Perform Welch's T-test (A vs Others)
            stat, _ = ttest_ind(emb[mask_target, d], emb[mask_others, d], equal_var=False)
            
            # Competitive selection: only keep dimensions where T > 0
            if stat > max_t:
                max_t = stat
                best_ct = ct
        
        if best_ct is not None:
            dim_records.append({'dim': d, 'assigned_ct': best_ct, 't_stat': max_t})
    
    df_dims = pd.DataFrame(dim_records)
    
    # Step 2: Extract Core Weights
    # For each cell type, calculate the mean attention weight across its top core dimensions
    plot_data = []
    for ct in unique_types:
        # Filter dimensions assigned to this cell type
        ct_dims = df_dims[df_dims['assigned_ct'] == ct].copy()
        if len(ct_dims) == 0:
            continue
        
        # Keep only the Top N most significant dimensions
        if len(ct_dims) > top_n_dims:
            ct_dims = ct_dims.sort_values('t_stat', ascending=False).head(top_n_dims)
        
        assigned_dims = ct_dims['dim'].values
        ct_mask = (labels == ct)
        
        # Calculate mean weight across assigned dimensions for each cell
        # Resulting shape: (num_cells_in_type,)
        w1_vals = alpha1[ct_mask][:, assigned_dims].mean(axis=1)
        w2_vals = alpha2[ct_mask][:, assigned_dims].mean(axis=1)
        
        # Build long-format records for visualization
        for v in w1_vals:
            plot_data.append({'CellType': ct, 'Weight': v, 'Omics': 'Omics1'})
        for v in w2_vals:
            plot_data.append({'CellType': ct, 'Weight': v, 'Omics': 'Omics2'})
            
    return pd.DataFrame(plot_data)

def select_resolution(reso_to_n_clusters, target):
    # 找到最小距离
    min_diff = min(abs(n - target) for n in reso_to_n_clusters.values())
    # 找出所有距离等于最小距离的 resolution
    candidates = [r for r, n in reso_to_n_clusters.items() if abs(n - target) == min_diff]

    # 分两类并选合适的
    lower = [r for r in candidates if reso_to_n_clusters[r] <= target]
    return max(lower) if lower else min(candidates)


def find_exact_resolution(target, adata_1, method, initial_reso, cluster_method="leiden",max_iter=10, step=0.01):
    sc.pp.neighbors(adata_1, use_rep="X_" + method)
    checked_resos = set()
    reso_to_clusters = {}
    initial_n_clusters = len(set(adata_1.obs[f"{cluster_method}_{method}_{initial_reso}"]))
    if initial_n_clusters > target:
        direction = -1
    else:
        direction = 1

    for i in range(max_iter):
        try_reso = round(initial_reso + direction * step * i, 2)
        if try_reso in checked_resos or try_reso <= 0:
            continue

        pred_key = f"{cluster_method}_{method}_{try_reso}"
        if cluster_method == "leiden":
            sc.tl.leiden(adata_1, resolution=try_reso, key_added=pred_key)
        elif cluster_method == "louvain":
            sc.tl.louvain(adata_1, resolution=try_reso, key_added=pred_key)
        n_clusters = len(set(adata_1.obs[pred_key]))
        reso_to_clusters[try_reso] = n_clusters
        checked_resos.add(try_reso)

        if n_clusters == target:
            return try_reso

    # 如果没找到精确匹配，则返回最接近目标的 resolution
    if reso_to_clusters:
        print("not target reolution", reso_to_clusters)
        closest_reso = min(reso_to_clusters, key=lambda r: abs(reso_to_clusters[r] - target))
        return closest_reso

    return None


def mclust_R(embedding, num_cluster, modelNames='EEE', random_seed=0):

    import numpy as np
    import rpy2.robjects as robjects
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import numpy2ri

    # 1. R 包
    robjects.r.library("mclust")

    # 2. seed
    robjects.r['set.seed'](random_seed)

    # 3. 安全转换 + 传数据（关键改动）
    with localconverter(default_converter + numpy2ri.converter):

        robjects.globalenv['X_input'] = embedding

        r_script = f"""
        data_mat <- as.matrix(X_input)
        res <- Mclust(data_mat, G={num_cluster}, modelNames="{modelNames}")
        res$classification
        """

        r_result = robjects.r(r_script)

    return np.array(r_result, dtype=int)


# def mclust_R(embedding, num_cluster, modelNames='EEE', random_seed=0):
    
#     """
#     通过直接执行 R 脚本字符串来运行 Mclust，
#     彻底避免 rpy2 函数调用时的参数/维度解析错误。
#     """
    
#     import numpy as np
#     import rpy2.robjects as robjects
#     from rpy2.robjects import numpy2ri
#     # 1. 激活 numpy 转换
#     numpy2ri.activate()
    
#     # 2. 加载 mclust 库
#     robjects.r.library("mclust")
    
#     # 3. 设置 R 种子
#     robjects.r['set.seed'](random_seed)
    
#     # ================= 核心修改 =================
#     # 策略：不直接调用函数，而是把数据“扔”进 R 的全局环境
    
#     # 步骤 A: 将 numpy 数组赋值给 R 全局变量 'X_input'
#     # numpy2ri 会自动将其转为 R 的矩阵/数组结构
#     robjects.globalenv['X_input'] = embedding
    
#     # 步骤 B: 编写一段纯 R 代码字符串
#     # 在 R 内部显式执行 as.matrix，确保万无一失
#     r_script = f"""
#     # 强制转换为矩阵，消除维度歧义
#     data_mat <- as.matrix(X_input)
    
#     # 执行聚类
#     res <- Mclust(data_mat, G={num_cluster}, modelNames="{modelNames}")
    
#     # 只返回分类结果
#     res$classification
#     """
    
#     # 步骤 C: 让 R 执行这段字符串
#     # 此时 R 是在自己的环境里运行，不会有 Python 传参的干扰
#     r_result = robjects.r(r_script)
#     # ===========================================
    
#     # 4. 转回 Numpy
#     return np.array(r_result).astype(int)



def dopca(X, dim=10):
    pcaten = PCA(n_components=dim, random_state=42)
    X_10 = pcaten.fit_transform(X)
    return X_10