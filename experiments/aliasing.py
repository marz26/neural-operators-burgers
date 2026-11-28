import numpy as np
N=16
x=np.linspace(0,2*np.pi,N,endpoint=False)
v=np.cos(6*x)
def aliasing(v):
    N=len(v)
    c=np.fft.fft(v)/N
    k=np.fft.fftfreq(N,1/N)
    for ki, ci in zip(k, c):
        if abs(ci) > 1e-10:
            print(ki, ci)

aliasing(v)
v=np.cos(6*x)**2
aliasing(v)
v=np.cos(16*x)
aliasing(v)
