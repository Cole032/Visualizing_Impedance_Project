import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Pth to data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Intrinsic_900_50%_60_circular_.csv")
save_path = os.path.join(script_dir, "acoustic_wave_90_20%.gif")

# Load the data
df = pd.read_csv(file_path, skiprows=1)

# speed of sound, density, and acoustic impedance of air
c_air = 343
rho_air = 1.21
Z_air = rho_air * c_air

L = 0.090  # Sample thickness (m)
D = 0.010  # Air gap depth (m)
mic1_x, mic2_x = -0.15, -0.05 

x_front = np.linspace(-0.5, 0, 300)
x_mat   = np.linspace(0, L, 200)
x_back  = np.linspace(L, L + D, 200)

fig, ax = plt.subplots(figsize=(12, 6))

# Animated Wave Lines
line_front, = ax.plot([], [], 'b-', lw=2, label='Front Tube (Air)')
line_mat,   = ax.plot([], [], 'r-', lw=2.5, label='Porous Sample')
line_back,  = ax.plot([], [], 'b-', lw=2, label='Back Cavity (Air Gap)')

# Envelope Lines in each region
env_f_up, = ax.plot([], [], 'k--', alpha=0.2)
env_f_dn, = ax.plot([], [], 'k--', alpha=0.2)
env_m_up, = ax.plot([], [], 'k--', alpha=0.2)
env_m_dn, = ax.plot([], [], 'k--', alpha=0.2)
env_b_up, = ax.plot([], [], 'k--', alpha=0.2)
env_b_dn, = ax.plot([], [], 'k--', alpha=0.2)

# Making sure the boundaries of each part of my impedance tube are clear
ax.axvspan(0, L, color='red', alpha=0.1)

ax.axvspan(L, L + D, color='blue', alpha=0.05)
ax.axvline(0, color='black', lw=2)
ax.axvline(L, color='black', linestyle='-.', lw=1.5)

ax.axvline(L + D, color='black', lw=4, label='Rigid Backing')
ax.plot([mic1_x, mic2_x], [0, 0], 'ko', markersize=6, label='Mics')

ax.set_xlim(-0.5, L + D + 0.05)
ax.set_ylim(-3.5, 3.5) 
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Acoustic Pressure")
ax.grid(True, alpha=0.3)
ax.legend(loc='lower left', fontsize=9)

title_text = ax.text(-0.48, 2.7, '', fontsize=10, family='monospace', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

current_time = 0.0
dt = 0.0001
frames_per_freq = 90 # If this is changed it changes how long the animation stays on one frequency
n_points = len(df)

def animate(frame):
    if frame % 10 == 0:
        print("At frame: " + str(frame))
    global current_time
    idx = (frame // frames_per_freq) % n_points
    row = df.iloc[idx]
    
    f     = row[0]
    Zc    = row[1] + 1j * row[2]
    gamma = row[3] + 1j * row[4]
    
    
    omega = 2 * np.pi * f
    k_air = omega / c_air
    current_time += dt 
    
    # Utsuno Math
    Z_back = -1j * Z_air / np.tan(k_air * D)
    term = np.tanh(gamma * L)
    Z_front = Zc * (Z_back + Zc * term) / (Zc + Z_back * term)
    R = (Z_front - Z_air) / (Z_front + Z_air)
    P_0, V_0 = 1 + R, (1 - R) / Z_air
    A2, B2 = 0.5 * (P_0 + Zc * V_0), 0.5 * (P_0 - Zc * V_0)
    P_L = A2 * np.exp(-gamma * L) + B2 * np.exp(gamma * L)
    V_L = (A2 * np.exp(-gamma * L) - B2 * np.exp(gamma * L)) / Zc
    A3, B3 = 0.5 * (P_L + Z_air * V_L), 0.5 * (P_L - Z_air * V_L)

   # These are the pressure fields described by the boundary conditions
    P_f = np.exp(-1j * k_air * x_front) + R * np.exp(1j * k_air * x_front)
    P_m = A2 * np.exp(-gamma * x_mat) + B2 * np.exp(gamma * x_mat)
    P_b = A3 * np.exp(-1j * k_air * (x_back - L)) + B3 * np.exp(1j * k_air * (x_back - L))
   #Updates envelope for in from of the medium
    env_f_up.set_data(x_front, np.abs(P_f))
    env_f_dn.set_data(x_front, -np.abs(P_f))
   #Update envelope inside the material
    env_m_up.set_data(x_mat, np.abs(P_m))
    env_m_dn.set_data(x_mat, -np.abs(P_m))
    #Update the envelope in the airspace ring
    env_b_up.set_data(x_back, np.abs(P_b))
    env_b_dn.set_data(x_back, -np.abs(P_b))

    # 2. Update Moving Waves for each new frequency
    t_f = np.exp(1j * omega * current_time)
    line_front.set_data(x_front, np.real(P_f * t_f))
    line_mat.set_data(x_mat, np.real(P_m * t_f))
    line_back.set_data(x_back, np.real(P_b * t_f))
    
    title_text.set_text(f"FREQ: {f:7.1f} Hz\n"
                        f"Zc:   {Zc.real:7.1f} + {Zc.imag:7.1f}j\n"
                        f"Gam:  {gamma.real:7.2f} + {gamma.imag:7.2f}j")

    return (line_front, line_mat, line_back, env_f_up, env_f_dn, 
            env_m_up, env_m_dn, env_b_up, env_b_dn, title_text)

ani = animation.FuncAnimation(fig, animate, frames = frames_per_freq*n_points, interval=30, blit=True) # if you want the whole thing do this frames_per_freq*n_points
#ani.save(save_path, writer='pillow', fps=30)
plt.show()
