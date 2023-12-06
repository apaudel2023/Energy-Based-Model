import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import os
import imageio.v2 as imageio
import pdb

# Generate 2D spiral data
def generate_spiral_data(num_samples=1000):
    t = np.linspace(0, 4 * np.pi, num_samples)
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.column_stack((x, y))
    return data

# Energy function for the model
def energy_function(x):
    return np.sum(x**2, axis=1)  # Euclidean norm as energy

# Langevin dynamics sampling
def langevin_dynamics(x, step_size, num_steps):
    for _ in range(num_steps):
        noise = np.random.normal(0, np.sqrt(2 * step_size), x.shape)
        x += step_size * energy_gradient(x) + noise
    return x

# Gradient of the energy function
def energy_gradient(x):
    return 2 * x

# Contrastive divergence training step
def contrastive_divergence_step(model, data, step_size=0.01, num_steps=100):
    positive_samples = data
    negative_samples = model.predict(data)
    
    # Update model parameters using CD
    model.train_on_batch(positive_samples, np.zeros_like(positive_samples))
    model.train_on_batch(negative_samples, np.ones_like(negative_samples))
    
    # Langevin dynamics for sampling
    negative_samples = langevin_dynamics(negative_samples, step_size, num_steps)
    model.train_on_batch(negative_samples, np.ones_like(negative_samples))

# Create and compile the model
model = Sequential()
model.add(Dense(256, input_shape=(2,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='linear'))

model.summary()
model.compile(optimizer='adam', loss='mse')
# pdb.set_trace()

# Training loop
num_epochs = 1000
batch_size = 32
result_loc = './../ebm3_results'
os.makedirs(result_loc, exist_ok=True)

for epoch in range(num_epochs):
    # Generate spiral data
    data = generate_spiral_data(num_samples=batch_size)

    # Update the model using contrastive divergence
    contrastive_divergence_step(model, data)

    # Plot and save samples after each epoch
    if epoch % 25 == 0:
        generated_samples = model.predict(np.random.normal(0, 1, (batch_size, 2)))
        plt.scatter(generated_samples[:, 0], generated_samples[:, 1], c='r', label='Generated Samples')
        plt.scatter(data[:, 0], data[:, 1], c='b', label='True Samples')
        plt.title(f'Epoch {epoch}')
        plt.legend()
        plt.savefig(result_loc+f'/samples_epoch_{epoch}.png')
        plt.close()


# Create a GIF from the saved figures
images = []
for epoch in range(0, num_epochs, 100):
    image_path = f'{result_loc}/samples_epoch_{epoch}.png'
    images.append(imageio.imread(image_path))

imageio.mimsave(result_loc+'/generated_samples.gif', images, duration=1)
                
