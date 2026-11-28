import numpy as np
from burgers.solver import spectral_derivative

def test_import():
    import burgers
    assert burgers is not None


def test_derivative_convergence():
    L = 2 * np.pi
    N=64
    x = np.linspace(0, L, N, endpoint=False)
    u = np.exp(np.sin(x))
    exact = np.cos(x) * np.exp(np.sin(x))
    err_spec = np.max(np.abs(spectral_derivative(u, L) - exact))
    assert err_spec < 1e-10, f"Errore derivata spettrale troppo grande: {err_spec:.2e}"


def test_derivative_const():
    L = 2 * np.pi
    N=64
    x = np.linspace(0, L, N, endpoint=False)
    u = np.ones_like(x)
    exact = np.zeros_like(x)
    err_spec = np.max(np.abs(spectral_derivative(u, L) - exact))
    assert err_spec < 1e-12, f"Errore derivata spettrale costante troppo grande: {err_spec:.2e}"


def test_second_derivative_convergence():
    L = 2 * np.pi
    N=64
    x = np.linspace(0, L, N, endpoint=False)
    u = np.exp(np.sin(x))
    exact = (np.cos(x)**2 - np.sin(x)) * np.exp(np.sin(x))
    err_spec = np.max(np.abs(spectral_derivative(u, L, order=2) - exact))
    assert err_spec < 1e-10, f"Errore derivata spettrale seconda troppo grande: {err_spec:.2e}"


def test_derivative_odd_N():
    L = 2 * np.pi
    N=63
    x = np.linspace(0, L, N, endpoint=False)
    u = np.sin(x)
    exact = np.cos(x)
    err_spec = np.max(np.abs(spectral_derivative(u, L, order=1) - exact))
    assert err_spec < 1e-10, f"Errore derivata spettrale con N dispari troppo grande: {err_spec:.2e}"

def test_third_derivative_convergence():
    L = 2 * np.pi
    N=64
    x = np.linspace(0, L, N, endpoint=False)
    u = np.sin(x)
    exact = -np.cos(x) # Terza derivata di sin(x)
    err_spec = np.max(np.abs(spectral_derivative(u, L, order=3) - exact))
    assert err_spec < 1e-10, f"Errore derivata spettrale terza troppo grande: {err_spec:.2e}"


def test_mask():
    from burgers.solver import dealias_mask
    L = 2 * np.pi
    N = 16
    mask = dealias_mask(N, L)
    assert mask.sum()== 11, "Maschera non corretta per frequenze basse"
    N = 15
    mask = dealias_mask(N, L)
    assert mask.sum() == 9, "Maschera non corretta per frequenze alte"

def test_nonlinear_hat():
    from burgers.solver import nonlinear_hat, dealias_mask, wavenumbers
    L = 2 * np.pi
    N = 16
    x = np.linspace(0, L, N, endpoint=False)
    u = np.sin(x)
    u_hat = np.fft.fft(u)
    k = wavenumbers(N, L)
    mask = dealias_mask(N, L)
    u2_hat_dealiased = nonlinear_hat(u_hat, k, mask)
    u2_dealiased = np.fft.ifft(u2_hat_dealiased).real
    u2_exact = -0.5*np.sin(2*x)
    err_nonlinear = np.max(np.abs(u2_dealiased - u2_exact))
    assert err_nonlinear < 1e-10, f"Errore nel calcolo nonlineare troppo grande: {err_nonlinear:.2e}"

def test_nonlinear_hat_mean():
    from burgers.solver import nonlinear_hat, dealias_mask, wavenumbers
    L = 2 * np.pi
    N = 16
    x = np.linspace(0, L, N, endpoint=False)
    u = np.sin(x)
    u_hat = np.fft.fft(u)
    k = wavenumbers(N, L)
    mask = dealias_mask(N, L)
    u2_hat_dealiased = nonlinear_hat(u_hat, k, mask)
    assert abs(u2_hat_dealiased[0]) < 1e-10, f"Errore nella media del calcolo nonlineare troppo grande: {u2_hat_dealiased[0]:.2e}"

def test_energy_mask():
    from burgers.solver import nonlinear_hat, dealias_mask, wavenumbers
    L = 2 * np.pi
    N = 16
    x = np.linspace(0, L, N, endpoint=False)
    u = np.sin(x)
    u_hat = np.fft.fft(u)
    k = wavenumbers(N, L)
    mask = dealias_mask(N, L)
    u_hat_dealiased = nonlinear_hat(u_hat, k, mask)
    assert abs((u*np.fft.ifft(u_hat_dealiased).real).sum()) < 1e-12, f"Errore nell'energia del calcolo nonlineare troppo grande: {abs(u*np.fft.ifft(u_hat_dealiased).real).sum():.2e}"

def test_energy_nomask():
    from burgers.solver import nonlinear_hat, wavenumbers
    L = 2 * np.pi
    N = 16
    x = np.linspace(0, L, N, endpoint=False)
    u = np.cos(6*x)+np.sin(4*x)
    u_hat = np.fft.fft(u)
    k = wavenumbers(N, L)
    u_hat_no_mask = nonlinear_hat(u_hat, k, np.ones_like(k))
    assert abs((u*np.fft.ifft(u_hat_no_mask).real).sum()) > 1e-15, f"Errore nell'energia del calcolo nonlineare senza maschera troppo piccolo: {abs(u*np.fft.ifft(u_hat_no_mask).real).sum():.2e}"