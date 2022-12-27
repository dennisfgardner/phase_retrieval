"""utilities"""

import numpy as np


def fft2(a):
    """shifted and normalized forward fourier transform"""
    try:
        M, N = a.shape
    except ValueError:
        print('fft2 expected a 2D array')
    A = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))/np.sqrt(M*N)
    return A


def ifft2(A):
    """shifted and normalized inverse fourier transform"""
    try:
        M, N = A.shape
    except ValueError:
        print('fft2 expected a 2D array')
    a = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A)))/np.sqrt(M*N)
    return a


def rect(x, D=1):
    """rectangle function as listed in "Numerical Simulation of Optical Wave
    Propagation" by Jason D. Schmidt

    there is a sampling check using the table by David Voelz in "computational
    fourier optics" pg. 18

    input:  x vector
            D diameter
    output: y vector
    """
    x = np.abs(x)
    y = np.copy(x)
    y = x < (D/2)
    y = y.astype(np.single)
    y[np.isclose(x, D/2)] = 0.5
    assert np.sum(y) >= 10, 'diameter of rect by be larger than 10'
    return y


def create_circle_px(radius, N=None):
    """return circle given radius in pixels within a grid of size N. N is
    assumed to be 2*radius if not given. If N is given, it must be greater than
    2*radius. It's also assumed the circle is at the center of grid.

    inputs:
        radius the radius of the circle in pixel units
        N the size of the grid, default 2*radius, must be >= 2*radius

    output:
        circle numpy array with zeros outside circle and ones inside

    This function is inspired by M. J. Townson, O. J. D. Farley, G. Orban de
    Xivry, J. Osborn, and A. P. Reeves, "AOtools: a Python package for adaptive
    optics modelling and analysis," Opt. Express 27, 31316-31329 (2019)
    """
    if N is None:
        N = 2*radius

    assert N >= 2*radius, 'gird size must be larger than 2*radius'

    x = np.arange(0.5, N, 1.0)
    # circle at center
    x = x - N/2

    [X, Y] = np.meshgrid(x, x)

    circle = np.zeros((N, N))
    mask = X**2 + Y**2 <= radius**2
    circle[mask] = 1

    return circle
