import numpy as np

def wavenumbers(N:int, L:float)->np.ndarray:
    k=np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    return k
def spectral_derivative(u: np.ndarray, L: float, order: int = 1) -> np.ndarray:
    N = len(u)
    k = wavenumbers(N, L)
    mult = (1j * k) ** order
    if order % 2 == 1 and N % 2 == 0:
        mult[N // 2] = 0.0              # modo di Nyquist: derivata dispari = 0
    du = np.fft.ifft(mult * np.fft.fft(u))
    return du.real                      # du.imag deve essere ~1e-15

def dealias_mask(N: int, L: float) -> np.ndarray:
    k = wavenumbers(N, L)
    mask = np.abs(k) <= (2.0 / 3.0) * (np.max(np.abs(k)))
    return mask

def nonlinear_hat(u_hat: np.ndarray, k: np.ndarray, mask: np.ndarray) -> np.ndarray:
    u_hat_dealiased = u_hat * mask
    u=np.fft.ifft(u_hat_dealiased)
    u_real = u.real
    u2 = u_real ** 2
    u2_hat = np.fft.fft(u2)*(-0.5j*k)
    u2_hat_dealiased = u2_hat * mask
    return u2_hat_dealiased