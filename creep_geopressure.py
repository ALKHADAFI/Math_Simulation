import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  


# Parameter model
Cc = 0.393  # Slope kurva kompresi normal
Cr = 0.0655  # Slope elastik
Ca = 0.0053  # Slope creep
t0 = 85 * 60  # Konstanta waktu (dalam detik)
k = 1e-15  # Permeabilitas (m^2)
mu = 1.31e-3  # Viskositas air (Pa.s)
rho_w = 1000  # Densitas air (kg/m^3)
g = 9.81  # Gravitasi (m/s^2)
sigma_v0 = 1e6  # Tegangan vertikal awal (Pa)

# Kondisi awal dan domain
z_max = 2000  # Kedalaman maksimum (m)
dz = 10  # Ukuran grid (m)
dt = 1e3  # Ukuran waktu (s)
time_steps = 1000  # Jumlah langkah waktu
n_cells = int(z_max / dz)  # Jumlah grid dalam domain

# Array awal
z = np.linspace(0, z_max, n_cells)
P = np.zeros(n_cells)  # Tekanan pori awal (hidrostatik)
sigma_v = rho_w * g * z + sigma_v0  # Tegangan vertikal
e = 0.85 - Cc * np.log(sigma_v / sigma_v0)  # Void ratio awal
porosity = e / (1 + e)  # Porositas awal

# Fungsi permeabilitas
def permeability(n):
    a, b = 10.6, -22.6
    return 10 ** (a * n + b)

# Iterasi numerik
for t in range(time_steps):
    P_new = np.copy(P)
    for i in range(1, n_cells - 1):
        # Hitung tegangan efektif
        effective_stress = sigma_v[i] - P[i]
        effective_stress = max(effective_stress, 1e-10)  # Hindari nilai negatif atau nol
        
        # Update void ratio
        e[i] = 0.85 - Cc * np.log(effective_stress / sigma_v0)
        
        # Update porositas
        porosity[i] = e[i] / (1 + e[i])
        
        # Perhitungan tekanan pori
        dP_dt = (
            effective_stress / (1 + porosity[i]) * Cr
            + Ca / (t0 * (1 + porosity[i])) * np.exp(-e[i] / Ca)
        )
        k_eff = permeability(porosity[i])
        dP_dz2 = (P[i+1] - 2*P[i] + P[i-1]) / dz**2
        dP_dt += (k_eff / mu) * dP_dz2
        
        # Update tekanan pori
        P_new[i] += dP_dt * dt
    
    # Kondisi batas
    P_new[0] = P[0]
    P_new[-1] = P[-2]
    
    # Update array tekanan pori
    P = np.copy(P_new)


# Visualisasi hasil
plt.plot(P / 1e6, z, label="Tekanan pori (MPa)")
plt.gca().invert_yaxis()
plt.xlabel("Tekanan pori (MPa)")
plt.ylabel("Kedalaman (m)")
plt.legend()
plt.show()