# Ideal Gas Simulation

import numpy as np
import matplotlib.pyplot as plt

class IdealGas:
    def __init__(self, n, V, T):
        self.n = n  # number of moles
        self.V = V  # volume in m^3
        self.T = T  # temperature in K
        self.R = 8.314  # ideal gas constant

    def pressure(self):
        return (self.n * self.R * self.T) / self.V

# Example usage
if __name__ == '__main__':
    gas = IdealGas(n=1, V=0.022414, T=273.15)
    print(f'Pressure: {gas.pressure()} Pa')
