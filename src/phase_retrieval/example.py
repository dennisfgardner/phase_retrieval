import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter


from utilities import fft2, ifft2
from iterative_algorithms import ER, GIF_RAAR


# test object
obj = mpimg.imread('src/phase_retrieval/test.png')
obj = obj[:, :, 3]
m, n = obj.shape
obj[obj < 0.5] = 0
obj[obj > 0] = 1
obj = np.pad(obj, (m*2-m//2, n*2-n//2))

M, N = obj.shape

# create support
support = gaussian_filter(obj, sigma=3)
support[support < 0.1] = 0
support = support.astype(bool)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(obj)
axs[1].imshow(support)
plt.show()

# generated diffraction pattern without phase, i.e., modulus
modulus = np.abs(fft2(obj))
M, N = modulus.shape
fft_scaling_factor = np.sqrt(M*N)

# number of iterations
num_its = 500

# initial object guess, modulus * phase guess
rho = ifft2(modulus*np.exp(1j*np.ones_like(modulus)))

plt.figure()
plt.imshow(np.abs(rho))
plt.show()

# for speed inside the phase retrieval algorithm
modulus = np.fft.fftshift(modulus)*fft_scaling_factor
for ii in range(num_its):
    if ii % 10 == 0:
        print(f'on iteration {ii}')
    rho1 = ER(rho, modulus, support, positivity=True)
    rho2 = GIF_RAAR(rho, modulus, support, positivity=True)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs[0, 0].imshow(obj)
axs[0, 1].imshow(np.abs(rho1))
axs[0, 1].imshow(np.abs(rho2))
plt.show()
