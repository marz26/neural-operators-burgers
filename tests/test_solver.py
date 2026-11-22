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
