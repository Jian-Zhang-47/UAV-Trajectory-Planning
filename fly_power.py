import numpy as np
from config import *

def fly_power(V=UAV_SPEED, U_tip=ROTOR_TIP_SPEED, d0=PARASITE_DRAG_COEFF, rho=AIR_DENSITY, 
              s=ROTOR_SOLIDITY, A=ROTOR_DISK_AREA, delta=PROFILE_DRAG_COEFF, 
              omega=BLADE_ANGULAR_VELOCITY, R=ROTOR_RADIUS, k=CORRECTION_FACTOR, 
              W=AIRCRAFT_WEIGHT):
    
    # Calculate the hover velocity based on aircraft weight and air density
    v0 = np.sqrt(W / (2 * rho * A))  # Hover velocity (m/s)
    
    # Calculate the power needed to overcome profile drag
    P0 = (delta / 8) * rho * s * A * omega**3 * R**3  # Profile drag power (W)
    
    # Calculate induced power required for flight
    Pi = (1 + k) * (W**(3/2)) / np.sqrt(2 * rho * A)  # Induced power (W)
    
    # Calculate the terms based on UAV speed and rotor properties
    term1 = P0 * (1 + 3 * V**2 / U_tip**2)  # Term 1: Profile drag with velocity adjustment
    term2 = Pi * np.sqrt(np.sqrt(1 + V**4 / (4 * v0**4)) - V**2 / (2 * v0**2))  # Term 2: Induced power correction
    term3 = 0.5 * d0 * rho * s * A * V**3  # Term 3: Parasite drag power
    
    # Return the total power required by summing all terms
    return term1 + term2 + term3
