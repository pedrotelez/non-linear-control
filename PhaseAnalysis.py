#%% [Markdown]
# Non-minimum phase system
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Create a transfer function G(s) = s + 2/(s^2 + 3s + 1)
num = [1, 2]
den = [1, 3, 1]
MINIMUM_PHASE = signal.TransferFunction(num, den) # minimum phase system

# create a non-minimum phase system by shiffing the zero to the left
num_2 = [-1, 2]
den_2 = [1, 3, 1]
NON_MINIMUM_PHASE = signal.TransferFunction(num_2, den_2)

# Plot the magnitude and phase response
w, mag, phase = signal.bode(MINIMUM_PHASE)
w_2, mag_2, phase_2 = signal.bode(NON_MINIMUM_PHASE)

plt.figure()
plt.semilogx(w, mag, 'r-',label='Minimum Phase System', linewidth=2)    # Bode magnitude plot
plt.semilogx(w_2, mag_2, 'b--',label='Non-Minimum Phase System', linewidth=1)    # Bode magnitude plot
plt.title('Bode magnitude plot')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.legend()

plt.figure()
plt.semilogx(w, phase, label='Minimum phase System')  # Bode phase plot
# plt.semilogx(w_2, phase_2, label='Non-Minimum Phase System')  # Bode phase plot
plt.title('Bode phase plot')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Phase [deg]')
plt.grid()
plt.legend()

# simulate the step response of the system
t, y = signal.step(MINIMUM_PHASE)
t_2, y_2 = signal.step(NON_MINIMUM_PHASE)

plt.figure()
plt.plot(t, y, label='Minimum Phase System')
plt.plot(t_2, y_2, label='Non-Minimum Phase System')
# plot the input signal u(t)
# plt.plot(t, 2*np.ones(len(t)), label='Input Signal')
plt.title('Step response')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()
plt.show()
