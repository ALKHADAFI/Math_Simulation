import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameter Model
Lx, Ly = 10, 10   # Ukuran area tanah (m)
dx, dy = 0.5, 0.5     # Resolusi grid (m)
dt = 0.1          # Langkah waktu (s)
T = 20            # Waktu total simulasi (s)

D0 = 0.05  # Koefisien difusi dasar (m²/s)
v0 = 0.02  # Kecepatan aliran air tanah dasar (m/s)
k = 0.001  # Laju degradasi limbah

Nx, Ny = int(Lx/dx), int(Ly/dy)  # Jumlah grid
Nt = int(T/dt)  # Jumlah langkah waktu

# Definisikan Porositas Tanah (Berbeda untuk Setiap Posisi)
porositas = np.random.uniform(0.3, 0.6, (Nx, Ny))  # Porositas antara 30%-60%
D = D0 * porositas  # Koefisien difusi bergantung pada porositas
v = v0 * porositas  # Kecepatan aliran bergantung pada porositas

# Tambahkan Sumber Air Tanah
sumber_air_x, sumber_air_y = Nx//4, Ny//2  # Lokasi sumber air (di kiri tengah)
v[sumber_air_x:, :] += 0.5  # Aliran lebih cepat ke kanan dari sumber

# Inisialisasi Konsentrasi Limbah
C = np.zeros((Nx, Ny))  
C[Nx//2, Ny//2] = 10  # Sumber limbah di tengah

# Setup Plot untuk Animasi
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(C, cmap="coolwarm", origin="lower", extent=[0, Lx, 0, Ly])
ax.set_xlabel("Posisi X (m)")
ax.set_ylabel("Posisi Y (m)")
ax.set_title("Penyebaran Limbah dalam Tanah dengan Sumber Air")

# Tambahkan vektor arah aliran air tanah
X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
v_x = np.ones((Nx, Ny)) * v  # Kecepatan aliran ke kanan
v_y = np.zeros((Nx, Ny))  # Tidak ada aliran vertikal
quiver = ax.quiver(X, Y, v_x.T, v_y.T, color='black', scale=5)

# Fungsi untuk Update Simulasi
def update(frame):
    global C
    C_new = C.copy()

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            advection_x = -v[i, j] * (C[i, j] - C[i-1, j]) / dx
            advection_y = -v[i, j] * (C[i, j] - C[i, j-1]) / dy
            diffusion_x = D[i, j] * (C[i+1, j] - 2*C[i, j] + C[i-1, j]) / dx**2
            diffusion_y = D[i, j] * (C[i, j+1] - 2*C[i, j] + C[i, j-1]) / dy**2
            C_new[i, j] = C[i, j] + dt * (advection_x + advection_y + diffusion_x + diffusion_y - k * C[i, j])
    
    C[:] = C_new[:]
    im.set_array(C)
    ax.set_title(f"Penyebaran Limbah dengan Sumber Air (t = {frame*dt:.1f} s)")
    return [im, quiver]

# Animasi
ani = animation.FuncAnimation(fig, update, frames=Nt, interval=50)
plt.show()