import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from keras.optimizers import Adam
from models import FCNet  # Assuming you have the FCNet model in your models.py file
from langevin import sample_langevin
from data import potential_fn

# Parse command line arguments
energy_function = 'u1'  # Change this if needed
no_arrow = False  # Set to True to disable display of arrows
output_file = None  # Set to a specific file name if needed

# Configuration
grid_lim = 4
n_grid = 100
n_sample = 100
stepsize = 0.03
n_steps = 100

# Prepare for contour plot
energy_fn = potential_fn(energy_function)

xs = np.linspace(-grid_lim, grid_lim, n_grid)
ys = np.linspace(-grid_lim, grid_lim, n_grid)
XX, YY = np.meshgrid(xs, ys)
grids = np.stack([XX.flatten(), YY.flatten()]).T
e_grid = energy_fn(grids)

# Run Langevin dynamics
grad_log_p = lambda x: -energy_fn(x)
x0 = np.random.randn(n_grid, 2)

# Model definition
model = FCNet(in_dim=2, out_dim=2)  # Modify this based on your FCNet architecture
optimizer = Adam(learning_rate=stepsize)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Langevin dynamics sampling
l_sample, l_dynamics = sample_langevin(x0, model, grad_log_p, stepsize, n_steps, intermediate_samples=True)

# Plotting
fig = plt.figure()
ax = plt.gca()
plt.axis('equal')

point = plt.scatter([], [])
if no_arrow:
    arrow = None
else:
    arrow = plt.quiver([0], [0], [1], [1], scale=0.5, scale_units='xy', headwidth=2, headlength=2, alpha=0.3)
plt.tight_layout()

def init():
    """initialize animation"""
    global point, arrow
    ax = plt.gca()
    ax.contour(XX, YY, np.exp(-e_grid.view(100, 100)))
    return point, arrow

def update(i):
    """update animation for i-th frame"""
    global point, arrow, ax
    g = l_dynamics[i]
    s = l_sample[i]

    point.set_offsets(s)
    if arrow:
        arrow.set_offsets(s)
        arrow.set_UVC(U=g[:, 0], V=g[:, 1])
    ax.set_title(f'Step: {i}')
    return point, arrow

# Animation
anim = FuncAnimation(fig, update, frames=np.arange(100),
                     init_func=init,
                     interval=200, blit=False)

if output_file is None:
    outfile = f'imgs/{energy_function}.gif'
else:
    outfile = f'imgs/{output_file}.gif'

anim.save(outfile, writer='pillow', dpi=80)
print(f'file saved in {outfile}')
