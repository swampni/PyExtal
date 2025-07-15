import numpy as np


def sgnoise(img, m, g, A, delta, varB):
    img[img<1] = 1
    DQE = 1 / (A + delta*img + varB/img)
    return m*g*img/DQE

def DQE(img, varB, delta, A):
    return 1 / (A + delta*img + varB/img)

def chisq2d(exp, convolved, bval, m, g, A, delta, varB):
    error = (convolved - exp)**2
    tden = sgnoise(convolved+bval, m, g, A, delta, varB)
    tchi = np.nansum(error[tden!=0]/tden[tden!=0])
    return tchi/(exp.shape[0]*exp.shape[1]-1)

def convolved2d(img, mtf):
    return np.real(np.fft.ifft2(np.fft.fft2(img)*mtf))


def lucy_Richardson(dp, mtf, background, niter, varB=0, delta=0, A=1, g=1, m=1, ):
    '''
    Lucy Richardson deconvolution

    args:
        dp: diffraction pattern
        mtf: modulation transfer function
        background: background of the diffraction pattern
        niter: number of iteration
        varB: variance of background
        delta: $\Delta$ is the squared percentage error in the gain image
        A: $1 + F + \frac{1}{mG}$ where F is the Fano noise, G is the number of well electrons produced by one incident electron
        F: Fano noise
        G: number of well electrons produced by one incident photon/electron
        g: gain
        m: mixing factor
    '''
    convar = m*g*varB
    exp = dp - background

    lucy = np.full(dp.shape, np.sum(exp)/(dp.shape[0]*dp.shape[1]), dtype=np.float32)
    lucyc = convolved2d(lucy, mtf)
    print(f"initial chisq: {chisq2d(exp, lucyc, background, m, g, A, delta, varB)}")

    for iter in range(niter):
        num = exp + convar + background
        den = lucyc + convar + background
        den[den == 0] = 1
        ratio = num/den

        lucy *= convolved2d(ratio, mtf)
        lucyc = convolved2d(lucy, mtf)

        print(f"iter: {iter} chisq: {chisq2d(exp, lucyc, background, m, g, A, delta, varB)}")
    return lucy

    