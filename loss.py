import warnings

import torch
import torch.nn as nn
from torch import Tensor
from utils import sigmoid, CosineSimilarityMatrix
from collections.abc import Callable

class DGCLoss(nn.Module):
    def __init__(self, k: float = 1e-3, penalize=False, normalize=True):
        super(DGCLoss, self).__init__()
        self.k = k
        self.penalize = penalize
        self.normalize = normalize
        self.similarity = CosineSimilarityMatrix()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return dgc_loss(input, target, self.k, self.penalize, self.normalize, self.similarity)

def dgc_loss(input: Tensor, target: Tensor, k: float = 1e-3, penalize: bool = False, normalize: bool = True, similarity: Callable = CosineSimilarityMatrix()) -> Tensor:

    # Similarity matrix
    sm = similarity(input, input)
    mask_diagonal = ~ torch.eye(sm.shape[0]).bool()
    ranking = sm[mask_diagonal].view(sm.shape[0], sm.shape[0]-1)

    # Ground-truth Ranking function
    gt = torch.abs(target.unsqueeze(0) - target.unsqueeze(1)).float()
    gt = gt[mask_diagonal].view(sm.shape[0], sm.shape[0]-1)

    indicator = ranking.unsqueeze(1) - ranking.unsqueeze(-1)
    # Indicator function
    # Assuming a perfect step function
    # indicator_true = (indicator > 0).float()
    # indicator = indicator_true.sum(-1) + 1

    # Smooth indicator function
    indicator = sigmoid(indicator, k=k)
    indicator = indicator.sum(-1) + .5

    # Relevance score
#    relevance = 10. / (gt + 1)
    relevance = 5-gt
    relevance = relevance.clamp(max=5, min=0)

    if penalize:
        relevance = relevance.exp2() - 1

    dcg = torch.sum(relevance / torch.log2(indicator + 1), dim=1)

    if not normalize:
        return -dcg.mean()

    relevance, _ = relevance.sort(descending=True)
    indicator = torch.arange(relevance.shape[-1], dtype=torch.float32, device=relevance.device)
    idcg = torch.sum(relevance / torch.log2(indicator + 2), dim=-1)

    ndcg = dcg / idcg
    return 1 - ndcg.mean()


class MAPLoss(nn.Module):
    def __init__(self, k: float = 1e-3):
        super(MAPLoss, self).__init__()
        self.k = k
        self.similarity = CosineSimilarityMatrix()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return map_loss(input, target, self.k, self.similarity)

def map_loss(input: Tensor, target: Tensor, k: float = 1e-8, similarity: Callable = CosineSimilarityMatrix()) -> Tensor:

    # Similarity matrix
    sm = similarity(input, input)
    mask_diagonal = ~ torch.eye(sm.shape[0]).bool()
    ranking = sm[mask_diagonal].view(sm.shape[0], sm.shape[0]-1)

    gt = target.unsqueeze(0) == target.unsqueeze(1)
    gt = gt[mask_diagonal].view(sm.shape[0], sm.shape[0]-1).float()

    indicator = ranking.unsqueeze(1) - ranking.unsqueeze(-1)

    # Indicator function
    # Assuming a perfect step function
    # indicator_true = (indicator > 0).float()
    indicator = sigmoid(indicator, k=k)

    accumulated_gt = (gt.unsqueeze(1)*indicator).sum(-1)
    # accumulated_gt_true = (gt.unsqueeze(1)*indicator_true).sum(-1) + gt
    indicator = indicator.sum(-1) + .5
    # indicator_true = indicator_true.sum(-1) + 1

    prec = accumulated_gt / indicator

    ap = torch.sum(gt*prec, dim=1)
    num_positives = gt.sum(-1)
    relevant = torch.where(num_positives != 0)

    ap = ap[relevant]
    num_positives = num_positives[relevant]
    ap = ap/num_positives
    return 1 - ap.mean()

