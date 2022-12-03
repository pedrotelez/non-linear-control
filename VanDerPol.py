#%% [Markdown] 
# Van der Pol Oscillator
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Setting parameters
mu: float = 1.0 
x_0: list[int] = [4, 6] # Initial conditions
T = np.linspace(0, 50, 1000)

# Defining the Van der Pol equation
def VanDerPol(X, t):
    x, y = X
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

# Solving the ODE
sol = odeint(VanDerPol, y0=x_0, t=T)

x_sol = [sol[i][0] for i in range(len(sol))]
y_sol = [sol[i][1] for i in range(len(sol))]

# plot the solution for different values of mu in the same figure
mu_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
for mu in mu_values:
    sol = odeint(VanDerPol, y0=x_0, t=T)
    x_sol = [sol[i][0] for i in range(len(sol))]
    y_sol = [sol[i][1] for i in range(len(sol))]
    plt.plot(x_sol, y_sol, label=f"μ = {mu}", linewidth=1)

plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.title('Van der Pol Oscillator')
plt.legend()

# create the images folder if it does not exist
if not os.path.exists('images/vanderpol'):
    os.makedirs('images/vanderpol')

# save the plot as a png file
plt.savefig('images/vanderpol/vanderpol.png')
print(f'>> Image saved in images/vanderpol folder')


#%% [Markdown]
# Inverted Pendulum on a Cart
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Setting parameters
m = 1.0 # mass of the cart
M = 5.0 # mass of the pendulum
b = 0.1 # friction coefficient
g = 9.81 # gravity
l = 1.0 # length of the pendulum
u = 0.0 # input force

# Initial conditions
x_0 = 0.0 # initial position of the cart
theta_0 = np.pi/2 # initial angle of the pendulum
x_dot_0 = 0.0 # initial velocity of the cart
theta_dot_0 = 0.0 # initial angular velocity of the pendulum

# Defining the ODE
def InvertedPendulum(X, t):
    x, theta, x_dot, theta_dot = X
    dxdt = x_dot
    dthetadt = theta_dot
    dx_dotdt = (u + m * l * theta_dot**2 * np.sin(theta) - m * g * np.sin(theta) * np.cos(theta) - b * x_dot) / (M + m * np.sin(theta)**2)
    dtheta_dotdt = (-u * np.cos(theta) - m * l * theta_dot**2 * np.sin(theta) * np.cos(theta) + (M + m) * g * np.sin(theta) - b * theta_dot * np.cos(theta)) / (l * (M + m * np.sin(theta)**2))
    return [dxdt, dthetadt, dx_dotdt, dtheta_dotdt]

# Solving the ODE
sol = odeint(InvertedPendulum, y0=[x_0, theta_0, x_dot_0, theta_dot_0], t=np.linspace(0, 10, 1000))

x_sol = [sol[i][0] for i in range(len(sol))]
theta_sol = [sol[i][1] for i in range(len(sol))]
x_dot_sol = [sol[i][2] for i in range(len(sol))]
theta_dot_sol = [sol[i][3] for i in range(len(sol))]

# Plot the results
plt.figure()
plt.plot(np.linspace(0, 10, 1000), x_sol, label='x(t)')
plt.plot(np.linspace(0, 10, 1000), theta_sol, label='theta(t)')
plt.title('Position and angle of the pendulum')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.figure()
plt.plot(np.linspace(0, 10, 1000), x_dot_sol, label='x_dot(t)')
plt.plot(np.linspace(0, 10, 1000), theta_dot_sol, label='theta_dot(t)')
plt.title('Velocity of the cart and pendulum')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

# create a gif of the simulation
import glob
import imageio
import os
from PIL import Image

# create a folder to store the images
if not os.path.exists('images'):
    os.makedirs('images')

# create a folder to store the gif
if not os.path.exists('gifs'):
    os.makedirs('gifs')

# save the images
for i in range(len(sol)):
    print(f'Saving image {i+1} of {len(sol)}')
    plt.figure()
    plt.plot(np.linspace(0, 10, 1000), x_sol, label='x(t)')
    plt.plot(np.linspace(0, 10, 1000), theta_sol, label='theta(t)')
    plt.title('Position and angle of the pendulum')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.plot([0, x_sol[i]], [0, l*np.sin(theta_sol[i])], 'r-')
    plt.plot(x_sol[i], l*np.sin(theta_sol[i]), 'ro')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.savefig('images/{}.png'.format(i))
    plt.close()

# create the gif
images = []
for filename in glob.glob('images/*.png'):
    print(f'Adding image {filename} to the gif')
    images.append(imageio.imread(filename))
    imageio.mimsave('gifs/InvertedPendulum.gif', images, duration=0.1)

    
print('Gif created')
# delete the images
for filename in glob.glob('images/*.png'):
    os.remove(filename)

# delete the folder
os.rmdir('images')


# # create a gif to visualize the motion of the pendulum
# import matplotlib.animation as animation
# from IPython.display import HTML

# fig = plt.figure()
# ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
# ax.grid()

# line, = ax.plot([], [], 'o-', lw=2)
# time_template = 'time = %.1fs'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# def init():
#     line.set_data([], [])
#     time_text.set_text('')
#     return line, time_text

# def animate(i):
#     thisx = [0, x_sol[i]]
#     thisy = [0, l*np.sin(theta_sol[i])]
#     line.set_data(thisx, thisy)
#     time_text.set_text(time_template % (i*0.01))
#     return line, time_text

# ani = animation.FuncAnimation(fig, animate, np.arange(1, len(sol)),
#                                 interval=25, blit=True, init_func=init)

# HTML(ani.to_html5_video())

# # show the animation
# plt.show()


#%% [Markdown]

