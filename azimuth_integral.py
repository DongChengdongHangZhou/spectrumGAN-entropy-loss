from scipy import ndimage
import numpy as np
import torch


# def GetPSD1D(psd2D):
#     psd2D = 0.5*psd2D + 0.5
#     psd2D = np.exp(psd2D*16)-1
#     h  = psd2D.shape[0]
#     w  = psd2D.shape[1]
#     wc = w//2
#     hc = h//2
#     # create an array of integer radial distances from the center
#     Y, X = np.ogrid[0:h, 0:w]
#     r    = np.hypot(X - wc, Y - hc).astype(np.int)
#     # SUM all psd2D pixels with label 'r' for 0<=r<=wc
#     # NOTE: this will miss power contributions in 'corners' r>wc
#     psd1D = ndimage.sum(psd2D, r, index=np.arange(0, wc))
#     psd1D = psd1D/psd1D[0]
#     return torch.from_numpy(psd1D)

def GetPSD1D(psd2D):
    psd2D = 0.5*psd2D + 0.5
    psd2D = torch.exp(psd2D*16)-1
    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:256, 0:256]
    r    = np.hypot(X - 128, Y - 128).astype(np.int)
    r = torch.from_numpy(r).cuda(0)
    result = torch.zeros((128,)).cuda(0)
    for i in range(128):
        get = torch.eq(r,i)
        result[i] = (psd2D*get).sum()

    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = (result/result[0]).cuda(0)
    return psd1D


def crossEntropyFourierAzimuth(tensor1,tensor2):

    '''
    you must rewrite your own crossEntropyLoss since
    the pytorch version of crossEntropyLoss is
    (-p(x)*log(q(x))).sum()
    but the crossEntropyLoss applied in this paper is
    (-p(x)*log(q(x))-(1-p(x))*log(1-q(x))).sum()
    for the crossEntropyLoss, the sequence of tensor1
    and tensor2 cannot be changed
    '''
    # tensor1 = tensor1[1:128]
    # tensor2 = tensor2[1:128]
    # loss = (-tensor1*torch.log(tensor2)-(1-tensor1)*torch.log(1-tensor2)).sum()/127
    loss = ((tensor1-tensor2)*(tensor1-tensor2)).sum()
    return loss


