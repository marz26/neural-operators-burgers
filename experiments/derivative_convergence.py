import numpy as np
from burgers.solver import spectral_derivative
from burgers.solver import wavenumbers
import matplotlib.pyplot as plt

L = 2 * np.pi
print(f"{'N':>4} {'err spettrale':>14} {'err diff. finite':>17}")
Ns = [8, 16, 32, 64]
errs_spec = []
errs_fd = []

for N in Ns:
  x = np.linspace(0, L, N, endpoint=False)
  u = np.exp(np.sin(x))
  exact = np.cos(x) * np.exp(np.sin(x))

  err_spec = np.max(np.abs(spectral_derivative(u, L) - exact))
  errs_spec.append(err_spec)

  h = L / N
  fd = (np.roll(u, -1) - np.roll(u, 1)) / (2 * h)
  err_fd = np.max(np.abs(fd - exact))
  errs_fd.append(err_fd)

  print(f"{N:>4} {err_spec:>14.2e} {err_fd:>17.2e}")

fig, ax = plt.subplots()
ax.semilogy(Ns, errs_spec, marker="o", label="Errore derivata spettrale")
ax.semilogy(Ns, errs_fd, marker="s", label="Errore derivata a differenze finite")
ax.set_xlabel("Numero di punti N")
ax.set_ylabel("Errore massimo (scala log)")
ax.set_title("Convergenza della derivata prima")
ax.legend()
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
fig.savefig("experiments/derivative_convergence.png", dpi=150, bbox_inches="tight")
plt.show()

def spectral_derivative_img(u: np.ndarray, L: float, order: int = 1) -> np.ndarray:
    N = len(u)
    k = wavenumbers(N, L)
    mult = (1j * k) ** order
    if order % 2 == 1 and N % 2 == 0:
        mult[N // 2] = 0.0              # modo di Nyquist: derivata dispari = 0
    du = np.fft.ifft(mult * np.fft.fft(u))
    return du.imag 

du_img=spectral_derivative_img(u, L)
print(f"du.imag: {np.max(np.abs(du_img)):.2e} (deve essere ~1e-15)")


