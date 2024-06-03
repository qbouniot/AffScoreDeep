import torch
from utils import histogram

def sparsity_calc(tensor, threshold=1e-3, split=64):
    """
    Calculate the sparsity of a 4D tensor
    Args:
        tensor (torch.Tensor): 4D tensor
    Returns:
        sparsity (float): sparsity of the input tensor
    """
    
    nnz = 0
    for i in range(0, tensor.size(0), split):
        batch = tensor[i:i+split]
        binary_mask = (batch.abs() > threshold).float()
        nnz += torch.sum(binary_mask)
    sparsity = 1 - (nnz / tensor.numel())
    return sparsity.numpy()


def entropy_calc(hgram, bw=1):
    '''entropy of per layer'''

    px = hgram/ torch.sum(hgram)
    nzs = px > 0
    
    return -(torch.sum(px[nzs] * torch.log(px[nzs])) + torch.log(1/torch.tensor(bw)))

def entropy_batch_calc(array_in):
    '''average entropy per layer per batch'''

    entropy_cumulative = 0

    dim = array_in.shape[1]
    bins = dim

    for image_in in array_in:
        hist_1d, _ = histogram(image_in, bins=bins)
        entropy_cumulative += entropy_calc(hist_1d, bins)
    return (entropy_cumulative/array_in.shape[0]).cpu().numpy()