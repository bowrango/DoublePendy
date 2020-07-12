import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Pendulum rod lengths and masses
L1 = 1                 # (kg)
L2 = 1                 # (m)
m1 = 1                 # (kg)
m2 = 1                 # (kg)

# Initial angles and angular velocities of the double pendulum
theta1 = np.pi/4     # (rad.)
theta1dot = 0          # (rad./s)

theta2 = np.pi/2    # (rad.)
theta2dot = 0          # (rad./s)

# Gravitational acceleration
g = 9.81               # (m/s^2)

# Animation properties: 15 seconds at 60 frames per second -> 900 images
DURATION = 15
FPS = 60
dt = 1 / FPS

# Define the time range for calculations
t = np.arange(0, DURATION+dt, dt)


def derivative(y, t, L1, L2, m1, m2):
    """Returns the first derivative of y = theta1, z1, theta2, z2"""

    theta1, z1, theta2, z2 = y

    cos, sin = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1

    z1dot = (m2*g*np.sin(theta2)*cos - m2*sin*(L1*z1**2*cos + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*sin**2)

    theta2dot = z2

    z2dot = ((m1+m2)*(L1*z1**2*sin - g*np.sin(theta2) + g*np.sin(theta1)*cos) +
             m2*L2*z2**2*sin*cos) / L2 / (m1 + m2*sin**2)

    return theta1dot, z1dot, theta2dot, z2dot


# Initial conditions of system
y0 = np.array([theta1, theta1dot, theta2, theta2dot])

# Numerically integrate the equations of motion
y = odeint(derivative, y0, t, args=(L1, L2, m1, m2))

# Grab the theta vectors from the main matrix
theta1 = y[:, 0]
theta2 = y[:, 2]

# Convert the two bob positions to Cartesian coordinates
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# Plotted bob circle radius
radius = 0.05

def make_frame(i):
    """Generates an image snapshot of the double pendulum corresponding to a time i"""

    # Plot the rods of the pendulums between each x and y point
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='grey')

    # Plot circles at the x and y points to represent the two masses, along with the fixed point
    c0 = Circle((0, 0), radius/2, fc='k', zorder=10)
    c1 = Circle((x1[i], y1[i]), radius, fc='k', ec='k', zorder=10)
    c2 = Circle((x2[i], y2[i]), radius, fc='k', ec='k', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # Fix the frame of the image to that it is centered on the fixed point
    ax.set_xlim(-L1-L2-radius, L1+L2+radius)
    ax.set_ylim(-L1-L2-radius, L1+L2+radius)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')

    # Save the image to the frames directory to be later used for animation
    plt.savefig('frames/_img{:04d}.png'.format(i), dpi=72)
    plt.cla()


# Animation figure properties: 6.25 x 6.25 inches at 300 pixels per inch.
fig = plt.figure(figsize=(6.25, 6.25), dpi=300)
ax = fig.add_subplot(111)

# Generate images and save to frames directory
for i in range(0, t.size - 1, 1):
    print(i, 'out of', DURATION*FPS)
    make_frame(i)

plt.figure()
plt.subplot(211)
plt.plot(t, x1)
plt.ylabel('x Position [m]')
plt.subplot(212)
plt.plot(t, y1)
plt.xlabel('Time [s]')
plt.ylabel('y Position [m]')

plt.suptitle('Trajectory of Bob 1 - Case 3')
plt.show()

plt.figure()
plt.subplot(211)
plt.plot(t, x2)
plt.ylabel('x Position [m]')
plt.subplot(212)
plt.plot(t, y2)
plt.xlabel('Time [s]')
plt.ylabel('y Position [m]')

plt.suptitle('Trajectory of Bob 2 - Case 2')
plt.show()

