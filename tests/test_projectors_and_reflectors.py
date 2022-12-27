import unittest

import numpy as np
import matplotlib.pyplot as plt

from src.phase_retrieval.utilities import rect, create_circle_px
from src.phase_retrieval.projectors_and_reflectors import Ps, Pm, Rs


PLOT = True


class test_projectors_and_reflectors(unittest.TestCase):

    def test_Ps(self):
        """test support projector"""
        N = 256
        x = np.arange(-N//2, N//2, 1)
        radius = 32
        support = create_circle_px(radius, N)
        rho = rect(x, 2*radius)
        rho = np.outer(rho, rho)
        rho[rho > 0] = 1
        rho = Rs(rho, support)

        if PLOT:
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            axs[0].imshow(rho)
            axs[0].set_title('rho')
            axs[1].imshow(support)
            axs[1].set_title('support')
            plt.show()

        self.assertAlmostEqual(np.sum(rho-support), 0)

    def test_Pm(self):
        """test modulus projector"""

        N = 256
        fft_scaling_factor = np.sqrt(N*N)

        # this simulates the measured diffraction pattern
        known_modulus = np.random.rand(N, N)
        known_modulus = np.fft.fftshift(known_modulus)*fft_scaling_factor

        # create an object
        rho = np.random.rand(N, N)*np.exp(1j*np.random.rand(N, N))

        # apply the modulus projector, which happens in frequency space
        rho_updated = Pm(rho, known_modulus)

        # after Pm update, go to frequency space and compare it's amplitude to
        # the known modulus
        rho_in_freq_space = np.fft.fft2(rho_updated)
        amplitude = np.abs(rho_in_freq_space)
        diff = np.abs(known_modulus - amplitude)

        # test passes is the sum of the difference is almost zero
        self.assertAlmostEqual(np.sum(diff), 0)

        if PLOT:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
            fig.suptitle('Test Modulus Projector')
            ax1.imshow(known_modulus)
            ax1.set_title('known modulus')
            ax2.imshow(amplitude)
            ax2.set_title('Amplitude After Projection')
            ax3.imshow(diff)
            ax3.set_title('Difference')
            plt.show()


if __name__ == '__main__':
    unittest.main()
