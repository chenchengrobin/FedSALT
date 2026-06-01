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
        'cosine': lambda s, t: cosine_similarity(s, t),
    }

    if mode not in dispatch:
        raise ValueError(f"Unsupported similarity mode: {mode}")

    return dispatch[mode](source, target)

# cosine similarity
def cosine_similarity(feature_src, feature_trg):
    src_norm = F.normalize(feature_src, dim=1)    
    trg_norm = F.normalize(feature_trg, dim=1)    
    sim_matrix = torch.mm(src_norm, trg_norm.t())       
    similarity = sim_matrix.mean(dim=1)                 

    return similarity.to(DEVICE)
