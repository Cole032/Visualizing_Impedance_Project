import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# System Values
f = 600
omega = 2 * np.pi * f
c_air = 343
rho_air = 1.21

Z_air = rho_air * c_air  # impedance of air
k_air = omega / c_air 

L = 0.20  # Thickness of porous sample
D = 0.05  # Depth of air cavity behind sample

# Mic Positions inside tube
mic1_x = -0.15 
mic2_x = -0.05

# Intrinsic Material Properties 
Zc = 800 - 400j        #Zc: Characteristic ipedance   
alpha = 15.0              
beta = k_air * 1.5        
gamma = alpha + 1j * beta    #gamma: propagation constant

# These are the boundary conditions (Impedance) based on utsuno's math
Z_back = -1j * Z_air / np.tan(k_air * D) #Impedance at the back of the medium
term = np.tanh(gamma * L)
Z_front = Zc * (Z_back + Zc * term) / (Zc + Z_back * term)  #Impedance at the front of the medium
R = (Z_front - Z_air) / (Z_front + Z_air)   #Reflection coefficient of material 

P_0  = 1 + R 
V_0 = (1 - R) / Z_air
A2 = 0.5 * (P_0 + Zc * V_0)

B2 = 0.5 * (P_0 - Zc * V_0)

P_L = A2 * np.exp(-gamma * L) + B2 * np.exp(gamma * L)
V_L = (A2 * np.exp(-gamma * L) - B2 * np.exp(gamma * L)) / Zc
A3 = 0.5 * (P_L + Z_air * V_L)
B3 = 0.5 * (P_L - Z_air * V_L)

# The three spaces within the impedance tube defined relative to the material
x_front = np.linspace(-0.6, 0, 300)
x_mat   = np.linspace(0, L, 200)
x_back  = np.linspace(L, L + D, 200)

#Pressure waves defined at each point inside of the rube
P_front_complex = np.exp(-1j * k_air * x_front) + R * np.exp(1j * k_air * x_front)
P_mat_complex   = A2 * np.exp(-gamma * x_mat) + B2 * np.exp(gamma * x_mat)
P_back_complex  = A3 * np.exp(-1j * k_air * (x_back - L)) + B3 * np.exp(1j * k_air * (x_back - L))

fig, ax = plt.subplots(figsize=(14, 6))
# Tthis shows the maximum value wave  that the data will oscilate between
ax.plot(x_front, np.abs(P_front_complex), 'k--', alpha=0.3)
ax.plot(x_front, -np.abs(P_front_complex), 'k--', alpha=0.3)
ax.plot(x_mat, np.abs(P_mat_complex), 'k--', alpha=0.3)
ax.plot(x_mat, -np.abs(P_mat_complex), 'k--', alpha=0.3)
ax.plot(x_back, np.abs(P_back_complex), 'k--', alpha=0.3)
ax.plot(x_back, -np.abs(P_back_complex), 'k--', alpha=0.3)

# for the legend (with some latex equations)
line_front, = ax.plot([], [], 'b-', lw=2, label='Front Tube (Air)')
line_mat,   = ax.plot([], [], 'r-', lw=2.5, label='Porous Sample ($Z_c$, $\gamma$)')
line_back,  = ax.plot([], [], 'g-', lw=2, label='Back Cavity (Air Gap)')


# Setting up microphone position
ax.plot([mic1_x, mic2_x], [0, 0], 'ko', markersize=8, label='Microphones')
ax.annotate('Mic 1', xy=(mic1_x, 0.2), ha='center')
ax.annotate('Mic 2', xy=(mic2_x, 0.2), ha='center')

# creating the visuaization for the regions within the impedance tube
ax.axvspan(0, L, color='red', alpha=0.1)
ax.axvspan(L, L + D, color='green', alpha=0.05)
ax.axvline(0, color='black', lw=2, label='Sample Face ($x=0$)')
ax.axvline(L, color='black', linestyle='-.', lw=2, label='Sample Rear ($x=L$)')
ax.axvline(L + D, color='black', lw=4, label='Rigid Backing ($x=L+D$)')
# Set the boundaries of out plot based on the boundaries of the tube

ax.set_xlim(-0.5, L + D + 0.05)
ax.set_ylim(-2.5, 2.5)
ax.set_title("Utsuno Method: Wave Propagation", fontsize=15, fontweight='bold')
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Acoustic Pressure")
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# create the animation 
def animate(frame):
    t = frame * 0.0001
    time_factor = np.exp(1j * omega * t)
    line_front.set_data(x_front, np.real(P_front_complex * time_factor))
    line_mat.set_data(x_mat, np.real(P_mat_complex * time_factor))
    line_back.set_data(x_back, np.real(P_back_complex * time_factor))
    return line_front, line_mat, line_back

ani = animation.FuncAnimation(fig, animate, frames=200, interval=20, blit=True)
plt.show()