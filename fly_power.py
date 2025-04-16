import numpy as np
from config import *
def fly_power(V=SPEED, U_tip=U_tip, d0=d0, rho=rho, s=s, A=A, delta=delta, omega=omega, R=R, k=k,W=W):
    v0 = np.sqrt(W/(2*rho*A))
    P0 = delta/8*rho*s*A*omega**3*R**3
    Pi = (1+k)*W**(3/2)/np.sqrt(2*rho*A)
    term1 = P0 * (1 + 3 * V**2 / U_tip**2)
    term2 = Pi * np.sqrt(np.sqrt(1 + V**4 / (4 * v0**4)) - V**2 / (2 * v0**2))
    term3 = 0.5 * d0 * rho * s * A * V**3
    return term1 + term2 + term3