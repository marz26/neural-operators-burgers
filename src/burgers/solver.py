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

