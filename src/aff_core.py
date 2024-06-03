import ot
from sklearn.covariance import LedoitWolf
import torch
import numpy as np
from scipy.linalg import sqrtm

#For two Gaussians N(m0,K0), N(m1,K1), compute the OT map F(x) = Ax + b, returning (A,b)

def torch_sqrtm(A, device='cuda'):
    A_ = A.to(torch.float64)
    e_vals, e_vecs = torch.linalg.eigh(A_)
    e_vecs = torch.real(e_vecs)
    e_vals = torch.real(e_vals)

    ### negative eigs
    idx = torch.argwhere(e_vals>0)
    if len(idx)!=len(e_vals):
        #print('Caution! Numerical errors in sqrtm with {}/{} negative eigs.'.format(len(e_vals)-len(idx), len(e_vals)))
        consts = torch.linspace(1e-7, 1e-6, len(e_vals)-len(idx), device=device)
        eval_pos = e_vals[idx].ravel()
        e_vals_new = torch.cat((consts, eval_pos)).ravel()
        if len(torch.unique(e_vals_new)) != len(e_vals):
            e_vals_new_unique = [e_vals_new[0]]
            for i in range(1, len(e_vals_new)):
                if e_vals_new[i] == e_vals_new[i-1]:
                    # print(e_vals_new[i])
                    e_vals_new_unique.append((e_vals_new[i+1] + e_vals_new[i-1])/2)
                else:
                    e_vals_new_unique.append(e_vals_new[i])
            e_vals_new = torch.FloatTensor(e_vals_new_unique)
            assert len(torch.unique(e_vals_new)) == len(e_vals)
    else:
        e_vals_new = e_vals
    Sigma = torch.sqrt(e_vals_new)
    
    return (e_vecs@torch.diag(Sigma)@e_vecs.T).to(torch.float32)


def OT_map(mX, covX, mY, covY, eps=1e-6, use_scipy=False):
    if torch.trace(covY) > 2*1e-6*covY.size(0):
        if use_scipy:
            A = sqrtm(covY)
        else:
            A = torch_sqrtm(covY)
        tmp = A@covX@A 
        if use_scipy:
            sqtmp = sqrtm(tmp)
        else:
            sqtmp = torch_sqrtm(tmp) 

        return (torch.real(A@torch.pinverse(sqtmp)@A), mY)
    else:
        return (torch.zeros_like(covY), mY)


#Apply an affine map by centering the input first
def Affine_map(X, A, b, mean):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    Y = X - mean
    return A@Y + b


#Transfer X [d x N_sim] dataset to Y [d x N_real].
#Returns AffMap(X), which is the transferrred dataset.
#Optionally also returns the AT map as a pair (A,b).

def sim2real(X, Y, correctCov=False, device='cuda:0', eps=1e-6):

    #Compute mean and covariance matrix for the simulated data
    mX = X.mean(1).reshape(-1,1)
    
    #Compute mean and covariance matrix for the real data
    mY = Y.mean(1).reshape(-1,1)

    if correctCov:
        shrink = LedoitWolf()
        CovX = torch.from_numpy(shrink.fit(X.cpu().numpy().T).covariance_).to(device) + 1e-6*torch.eye(X.shape[0], device=device)
        CovY = torch.from_numpy(shrink.fit(Y.cpu().numpy().T).covariance_).to(device) + 1e-6*torch.eye(Y.shape[0], device=device)
    else:
        CovX = torch.cov(X) + 1e-6*torch.eye(X.shape[0], device=device) #the last term is no avoid singularities
        CovY = torch.cov(Y) + 1e-6*torch.eye(Y.shape[0], device=device)
    

    #Compute the OT map from X to Y
    A, b = OT_map(mX, CovX, mY, CovY, eps)

    AffMap = lambda X: Affine_map(X, A, b, mX)

    return AffMap(X), CovY


def rho_aff(X, Y, correctCov=False, device='cuda', eps=1e-6):
    
    X2Y, CovY = sim2real(X, Y, correctCov=correctCov, device=device, eps=eps)

    nX = X.shape[1]
    nY = Y.shape[1]

    C = torch.cdist(X2Y.T, Y.T)**2

    #print(C.shape)
    mu_X = torch.ones(nX)/nX
    mu_Y = torch.ones(nY)/nY
    ot_cost = ot.emd2(mu_X, mu_Y, C, numItermax=1e7)
    upper_bound = 2*torch.trace(CovY)

    return (1 - torch.sqrt(ot_cost/upper_bound)).cpu().numpy()

