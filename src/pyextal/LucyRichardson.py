import numpy as np


def sgnoise(img, m, g, A, delta, varB):
    """Calculates the signal-dependent noise.

    Args:
        img (np.ndarray): The input image.
        m (float): Mixing factor.
        g (float): Gain.
        A (float): A parameter related to Fano noise and conversion gain.
        delta (float): Squared percentage error in the gain image.
        varB (float): Variance of the background.

    Returns:
        np.ndarray: The calculated signal-dependent noise.
    """
    img[img < 1] = 1
    DQE = 1 / (A + delta*img + varB/img)
    return m*g*img/DQE


def DQE(img, varB, delta, A):
    """Calculates the Detective Quantum Efficiency (DQE).

    Args:
        img (np.ndarray): The input image.
        varB (float): Variance of the background.
        delta (float): Squared percentage error in the gain image.
        A (float): A parameter related to Fano noise and conversion gain.

    Returns:
        np.ndarray: The calculated DQE.
    """
    return 1 / (A + delta*img + varB/img)


def chisq2d(exp, convolved, bval, m, g, A, delta, varB):
    """Calculates the 2D chi-squared value.

    Args:
        exp (np.ndarray): The experimental data.
        convolved (np.ndarray): The convolved model.
        bval (np.ndarray): The background values.
        m (float): Mixing factor.
        g (float): Gain.
        A (float): A parameter related to Fano noise and conversion gain.
        delta (float): Squared percentage error in the gain image.
        varB (float): Variance of the background.

    Returns:
        float: The calculated chi-squared value.
    """
    error = (convolved - exp)**2
    tden = sgnoise(convolved+bval, m, g, A, delta, varB)
    tchi = np.nansum(error[tden != 0]/tden[tden != 0])
    return tchi/(exp.shape[0]*exp.shape[1]-1)


def convolved2d(img, mtf):
    """Performs a 2D convolution using FFT.

    Args:
        img (np.ndarray): The input image.
        mtf (np.ndarray): The modulation transfer function.

    Returns:
        np.ndarray: The convolved image.
    """
    return np.real(np.fft.ifft2(np.fft.fft2(img)*mtf))


def lucy_Richardson(dp, mtf, background, niter, varB=0, delta=0, A=1, g=1, m=1,):
    """Performs Lucy-Richardson deconvolution.

    Args:
        dp (np.ndarray): The diffraction pattern.
        mtf (np.ndarray): The modulation transfer function.
        background (np.ndarray): The background of the diffraction pattern.
        niter (int): The number of iterations.
        varB (float, optional): Variance of the background. Defaults to 0.
        delta (float, optional): The squared percentage error in the gain image.
            Defaults to 0.
        A (float, optional): A parameter related to Fano noise and conversion
            gain, given by :math:`1 + F + \\frac{1}{mG}`. Defaults to 1.
        g (float, optional): The gain. Defaults to 1.
        m (float, optional): The mixing factor. Defaults to 1.

    Returns:
        np.ndarray: The deconvolved image.
    """
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

