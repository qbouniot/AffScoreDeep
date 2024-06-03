import torch

def centering(K, device='cuda'):
    n = K.shape[0]
    unit = torch.ones([n, n])
    I = torch.eye(n)
    H = (I - unit / n).to(device)

    return H@K@H  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return torch.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = X@X.T
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        if mdist < 0:
            print('Error in automatic sigma calc, using 0.1 as default.')
            sigma = .1
        else:
            sigma = torch.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma, device='cuda'):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y, device='cuda'):
    L_X = X@X.T
    #print(L_X.shape)
    L_Y = Y@Y.T
    return torch.sum(centering(L_X, device=device) * centering(L_Y, device=device))


def linear_CKA(X, Y, device='cuda'):
    hsic = linear_HSIC(X, Y, device=device)
    var1 = torch.sqrt(linear_HSIC(X, X, device=device))
    var2 = torch.sqrt(linear_HSIC(Y, Y, device=device))

    return (hsic / (var1 * var2)).cpu().numpy()


def kernel_CKA(X, Y, sigma=None, device='cuda'):
    hsic = kernel_HSIC(X, Y, sigma, device=device)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma, device=device))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma, device=device))

    return hsic / (var1 * var2)


if __name__=='__main__':

    X = torch.randn(1000, 200)
    Y = torch.randn(1000, 200)

    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

    print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
    print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))