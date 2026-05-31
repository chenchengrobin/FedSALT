import numpy as np
import torch.nn.functional as F

from src.config import *
import json

def active_select(clients, top_k=0.1):
    selected_clients = clients
    N = len(selected_clients[0].logit)

    # collect client similarity 、logit and sample number
    client_similarity = torch.stack([client.similarity for client in selected_clients], dim=1)  # (N, K)
    client_logit = torch.stack([client.logit for client in selected_clients], dim=1)  # (N, K, CLASS)
    client_samples = torch.tensor([len(client.dataset) for client in selected_clients]).unsqueeze(0).expand(N, -1)  # (N, K)
    client_id = torch.tensor([client.client_id for client in selected_clients]).unsqueeze(0).expand(N, -1)  # (N, K)

    #  change device
    client_logit = client_logit.to(DEVICE)
    client_similarity = client_similarity.to(DEVICE)
    client_samples = client_samples.to(DEVICE)
    client_id = client_id.to(DEVICE)

    # server data number  and select number
    select_num = max(1, int(len(clients) * top_k))

    # the first step select
    # select topk clients for each data by similarity
    _, topk_indices = torch.topk(client_similarity, k=select_num, largest=True)  # (N, select_num)
    data_indices = torch.arange(N).unsqueeze(1)  # (N, 1)
    logits = client_logit[data_indices, topk_indices, :]  # (N, K, CLASS)
    similarity = client_similarity[data_indices, topk_indices]  # (N, K)
    samples = client_samples[data_indices, topk_indices]  # (N, K)
    id = client_id[data_indices, topk_indices]  # (N, K)

    # the second step select
    # committee filtering by majority pseudo label
    pred_labels = torch.argmax(logits, dim=2)  # (N, select_num)
    pred_one_hot = F.one_hot(pred_labels, num_classes=10).float() # (N, select_num, CLASS)
    counts = pred_one_hot.sum(dim=1)  # (N, CLASS)
    majority_labels = torch.argmax(counts, dim=1)  # (N, )

    # masking
    mask = (pred_labels == majority_labels.unsqueeze(1))  # (N, select_num)

    # logit aggregation by consensus
    aggregated_logit = aggregated_by_consensus(logits=logits, similarity=similarity, mask=mask, simple_size=samples) 

    return aggregated_logit


def aggregated_by_consensus(logits, mask, similarity, simple_size, eps=1e-8):
    mask = mask.float() # (N, select_num)

    mask_logits = logits * mask.unsqueeze(2)  # (N, select_num, C)
    mask_similarity = similarity * mask  # (N, select_num)
    mask_simple = simple_size * mask # (N, select_num)

    denom_simple = mask_simple.sum(dim=1, keepdim=True)
    base_alpha = mask_simple / (denom_simple + eps)  # (N, select_num)

    mc = compute_margin_contributed(mask_similarity, mask_simple)  # (N, select_num)
    denom_mc = mc.sum(dim=1, keepdim=True) # (N, 1)
    sim_alpha = mc / (denom_mc + eps)  # (N, select_num)

    alpha = (1 - base_alpha) * sim_alpha  # (N, select_num)
    alpha = alpha / (alpha.sum(dim=1, keepdim=True) + eps)
    aggregated_logit = (mask_logits * alpha.unsqueeze(2)).sum(dim=1) # (N, C)
    return aggregated_logit


def compute_margin_contributed(similarity, simple_size):
    N, K = similarity.shape
    mc = similarity * simple_size
    mc_n = mc.sum(dim=1)  # (N, )

    exclude_mask = 1 - torch.eye(K, dtype=torch.float32, device=similarity.device) # (K, K)
    exclude_mask = exclude_mask.expand(N, -1, -1) # (N, K, K)

    similarity_expand = similarity.unsqueeze(1).expand(-1, K, -1) # (N, K, K)
    simple_expand = simple_size.unsqueeze(1).expand(-1, K, -1) # (N, K, K)
 
    masked_similarity = similarity_expand * exclude_mask
    masked_simple = simple_expand * exclude_mask
    mc_without_k = (masked_similarity * masked_simple).sum(dim=2) # (N, K)

    mc = mc_n.unsqueeze(1) - mc_without_k # (N, K)
    return mc

