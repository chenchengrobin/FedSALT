import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.config import *
BATCH_SIZE = 1024


def get_similarity(source, target, mode='euclidean'):
    source = _to_tensor(source)
    target = _to_tensor(target)

    dispatch = {
        'euclidean': lambda s, t: euclidean_similarity(s, t),
        'mahalanobis': lambda s, t: mahalanobis_distance(s, t),
        'kde': lambda s, t: kde_similarity(s, t),
        'cosine': lambda s, t: cosine_similarity(s, t),
    }

    if mode not in dispatch:
        raise ValueError(f"Unsupported similarity mode: {mode}")

    return dispatch[mode](source, target)


def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(DEVICE)
    if isinstance(x, torch.Tensor):
        return x.float().to(DEVICE)
    raise TypeError("Input must be numpy array or torch tensor")


def _batch_apply(func, src, trg):
    outs = []
    for i in range(0, len(src), BATCH_SIZE):
        outs.append(func(src[i:i + BATCH_SIZE], trg))
    return torch.cat(outs, dim=0)


# 余弦相似度
def cosine_similarity(feature_src, feature_trg):
    src_norm = F.normalize(feature_src, dim=1)    
    trg_norm = F.normalize(feature_trg, dim=1)    
    sim_matrix = torch.mm(src_norm, trg_norm.t())       
    similarity = sim_matrix.mean(dim=1)                 

    return similarity.to(DEVICE)


def euclidean_similarity(feature_src, feature_trg):
    feature_src = F.normalize(feature_src, dim=1)
    feature_trg = F.normalize(feature_trg, dim=1)

    def _compute(batch, target):
        dist = torch.cdist(batch, target)
        return -dist.min(dim=1)[0]   # 越大越相似

    return _batch_apply(_compute, feature_src, feature_trg)


def mahalanobis_distance(feature_src, feature_trg):
    feature_src = feature_src.float()
    feature_trg = feature_trg.float()

    target_centered = feature_trg - feature_trg.mean(dim=0)
    cov = torch.cov(target_centered.T) + 1e-6 * torch.eye(feature_trg.shape[1], device=DEVICE)
    inv_cov = torch.linalg.pinv(cov)

    delta = feature_src - feature_trg.mean(dim=0)
    result = torch.einsum('ni,ij,nj->n', delta, inv_cov, delta).sqrt()
    return -result.to(DEVICE)


def kde_similarity(feature_src, feature_trg, pca_dim=32):
    src = feature_src.detach().cpu().numpy()
    trg = feature_trg.detach().cpu().numpy()

    pca = PCA(n_components=min(pca_dim, trg.shape[1]))
    trg_pca = pca.fit_transform(trg)
    src_pca = pca.transform(src)

    scaler = StandardScaler()
    trg_pca = scaler.fit_transform(trg_pca)
    src_pca = scaler.transform(src_pca)

    kde = KernelDensity(bandwidth='scott')
    kde.fit(trg_pca)

    score = kde.score_samples(src_pca)
    return torch.from_numpy(score).float().to(DEVICE)


if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.randn(1000, 128)
    b = np.random.randn(300, 128)

    sim_euc = GetSimilarity(a, b, mode='euclidean')
    sim_mah = GetSimilarity(a, b, mode='mahalanobis')

    print(sim_euc.shape, sim_cls.shape, sim_mah.shape)
