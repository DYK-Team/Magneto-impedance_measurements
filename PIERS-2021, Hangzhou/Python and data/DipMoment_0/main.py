#
#
# In zero approximation, the program calculates the current distribution and the dipole moment in a wire inclusion
# (l - length, a - radius) with the impedance boundary conditions on its surface and surrounded
# with a medium with epsilon and mu.
#
# Also, in zero approximation, the program calculates the dipole moment of a copper wire
#
# Yujie Zhao @ March 2021
#
# University of St. Andrews, Fife, Scotland
#

import pandas as pd
import numpy as np
import scipy
from scipy import integrate

# Wire and material parameters
epsilon = complex(4.0, 0.0)  # effective dielectric constant of the matrix, (e1,e2)
mu = complex(1.0, 0.0)  # effective magnetic constant of the matrix, (mu1,mu2)
l = 2.5  # wire length in cm
a = 9.3e-4  # wire radius in cm

# Some constants
sigma = 5.4E17  # copper conductivity (Hz) in cgs
c = 2.9979250e+10  # speed of light in cm/s (cgs system)
i = complex(0.0, 1.0)  # imaginary unit
unit = complex(1.0, 0.0)  # real unit in the complex form
e0 = 1.0 / c  # electric field amplitude

filename = 'Field5.5Oe_ZeroStrain.csv'  # csv file for the experimental impedance dispersion (freq, real, imag)

df = pd.read_csv(filename, sep = ',', header = None)  # reading the impedance file
freq = df[0]  # frequency array
NFREQ = len(df[0])  # number of frequency points
# Recalculating the experimental impedance (Ohms) into the surface impedance (cgs)
Impedance = (df[1] + df[2] * 1j) * 1.0e+9 * (a / l) / (2.0 * c)

x1 = -l / 2.0  # wire left end
x2 = l / 2.0  # wire right end

# General Green's functions G(r) and Gf(a)
def G(r):
    return np.exp(-i * k * r) / (4.0 * np.pi * r)

def Gf(r):
    return (a ** 2) * (unit + i * k * r) * np.exp(-i * k * r) / (2.0 * (r ** 3))

# Function for calculating Q and Qf factors
def FQ(x):
    r = (x ** 2 + a ** 2) ** 0.5
    return (G(r)).real

def FQf(x):
    r = (x ** 2 + a ** 2) ** 0.5
    return (Gf(r)).real

# Current distribution; zero approximation
def j0(x):
    current = (i * f * epsilon * e0 / (2.0 * Q * (kn ** 2))) * (np.cos(kn * x) - np.cos(kn * l / 2.0)) \
              / np.cos(kn * l / 2.0)
    return current

def Re_j0(x):
    return (j0(x)).real

def Im_j0(x):
    return (j0(x)).imag

def moment():
    Integral = list(integrate.quad(Re_j0, x1, x2))
    Result = Integral[0]
    dip0 = unit * Result
    Integral = list(integrate.quad(Im_j0, x1, x2))
    Result = Integral[0]
    dip0 = - (dip0 + i * Result) * i / (2.0 * np.pi * f)
    return dip0

dip0_real = []  # real part of the dipole moment in zero approximation
dip0_imag = []  # imaginary part of the dipole moment in zero approximation
dip0copper_real = []  # real part of the dipole moment of the copper wire  in zero approximation
dip0copper_imag = []  # imaginary part of the dipole moment of the copper wire  in zero approximation

for n in range(NFREQ):
    f = freq[n]
    Z = Impedance[n]

    k = 2.0 * np.pi * f * (epsilon * mu) ** 0.5 / c

    Integral = list(integrate.quad(FQ, x1, x2))
    Q = Integral[0]
    Integral = list(integrate.quad(FQf, x1, x2))
    Qf = Integral[0]

    kn = k * (unit - i * c * Z * Qf / (4.0 * (np.pi ** 2) * a * f * mu * Q)) ** 0.5

    M = moment()
    dip0_real.append(c * M.real)
    dip0_imag.append(-c * M.imag)

    # Surface impedance of copper wire
    Z = (unit - i) * 2.0 * np.pi * (f * sigma)**0.5 * a / c
    Z = (unit - i) * (f * sigma)**0.5 * scipy.special.jv(0, Z) / (scipy.special.jv(1, Z) * 2.0 * sigma)
    kn = k * (unit - i * c * Z * Qf / (4.0 * (np.pi ** 2) * a * f * mu * Q)) ** 0.5
    M = moment()
    dip0copper_real.append(c * M.real)
    dip0copper_imag.append(-c * M.imag)

    print('frequency point = ', n + 1)

disper0 = np.column_stack((freq, dip0_real, dip0_imag))
disper0copper = np.column_stack((freq, dip0copper_real, dip0copper_imag))

np.savetxt('disper0.csv', disper0)
np.savetxt('disper0copper.csv', disper0copper)
