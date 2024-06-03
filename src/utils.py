import torch
import numpy
import random


def init_dict(dict=None):

    flag = False

    keys = ['aff_score', 'aff_score_corrected', 'cka_score', 'ent_before', 
                'ent_after', 'sparsity_before', 'sparsity_after', 'norm_diff', 'input2d', 'output2d', 'input_all', 'output_all', 'r2_score']
    
    if dict is None:
        flag= True
        dict = {}
        
        for key in keys:
            dict[key] = []
    else:

        for layer in dict:
            for key in keys:
                dict[layer][key] = []

    if flag:
        return dict

def set_seeds(seed=987):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def histogram(xs, bins):
    # Like torch.histogram, but works with cuda
    min, max = xs.min(), xs.max()
    counts = torch.histc(xs, bins, min=min, max=max)
    boundaries = torch.linspace(min, max, bins+1)
    return counts, boundaries

def shrinkage(returns):
    """Shrinks sample covariance matrix towards constant correlation unequal variance matrix.
    Ledoit & Wolf ("Honey, I shrunk the sample covariance matrix", Portfolio Management, 30(2004),
    110-119) optimal asymptotic shrinkage between 0 (sample covariance matrix) and 1 (constant
    sample average correlation unequal sample variance matrix).

    :param returns:
        t, n - returns of t observations of n shares.
    :return:
        Covariance matrix, sample average correlation, shrinkage.
    """
    t, n = returns.shape
    mean_returns = torch.mean(returns, axis=0, keepdims=True)
    returns -= mean_returns
    sample_cov = torch.transpose(returns, 0, 1) @ returns / t

    # sample average correlation
    var = torch.diag(sample_cov).reshape(-1, 1)
    sqrt_var = var ** 0.5
    unit_cor_var = sqrt_var * torch.transpose(sqrt_var, 0, 1)
    average_cor = ((sample_cov / unit_cor_var).sum() - n) / n / (n - 1)
    prior = average_cor * unit_cor_var

    prior[range(len(prior)), range(len(prior))] = var.squeeze()

    # pi-hat
    y = returns ** 2
    phi_mat = (torch.transpose(y, 0, 1) @ y) / t - sample_cov ** 2
    phi = phi_mat.sum()

    # rho-hat
    theta_mat = (torch.transpose(returns**3, 0, 1) @ returns) / t - var * sample_cov
    theta_mat.fill_diagonal_(0)

    rho = (
        torch.diag(phi_mat).sum()
        + average_cor * (1 / sqrt_var @ torch.transpose(sqrt_var, 0, 1) * theta_mat).sum()
    )

    # gamma-hat
    gamma = torch.linalg.norm(sample_cov - prior, "fro") ** 2

    # shrinkage constant
    kappa = (phi - rho) / gamma
    shrink = max(0, min(1, kappa / t))

    # estimator
    sigma = shrink * prior + (1 - shrink) * sample_cov

    return sigma