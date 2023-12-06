## Standard libraries
import os
import json
import math
import numpy as np
import random
import pdb

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['savefig.format'] = 'png'
# plt.rcParams['savefig.transparent'] = True
from PIL import Image

import seaborn as sns
sns.reset_orig()

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset
import torch.optim as optim

# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/tutorial8"

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# ===========================
import torch
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np

class SpiralDataset(Dataset):
    def __init__(self, num_samples=1000):
        theta = np.linspace(0, 5 * np.pi, num_samples)
        radius = theta
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        data = np.vstack((x, y)).T

        # Scale the entire dataset between -1 and 1
        data_min, data_max = data.min(axis=0), data.max(axis=0)
        data_scaled = 2 * (data - data_min) / (data_max - data_min) - 1
        self.data = torch.from_numpy(data_scaled).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample, 0  

# Create a synthetic spiral dataset
spiral_dataset = SpiralDataset(num_samples=5000)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(spiral_dataset))
val_size = len(spiral_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(spiral_dataset, [train_size, val_size])

# Define data loaders
train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
val_loader = data.DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False)

# ==========================================
class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class CNNModel(nn.Module):

    def __init__(self, hidden_features=32, out_dim=1, **kwargs):
        super().__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        c_hid1 = hidden_features//2
        c_hid2 = hidden_features
        c_hid3 = hidden_features*2

        # Series of convolutions and Swish activation functions
        self.cnn_layers = nn.Sequential(
                nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4), # [16x16] - Larger padding to get 32x32 image
                Swish(),
                nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1), #  [8x8]
                Swish(),
                nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1), # [4x4]
                Swish(),
                nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1), # [2x2]
                Swish(),
                nn.Flatten(),
                nn.Linear(c_hid3*4, c_hid3),
                Swish(),
                nn.Linear(c_hid3, out_dim)
        )

    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        return x

class PointModel(nn.Module):
    def __init__(self, input_dim=2, hidden_features=32, out_dim=1):
        super(PointModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_features*4),
            nn.ReLU(),
            nn.Linear(hidden_features*4, hidden_features*2),
            nn.ReLU(),
            nn.Linear(hidden_features*2, hidden_features ),
            nn.ReLU(),
            nn.Linear(hidden_features , hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, out_dim)
        )

    def forward(self, x):
        return self.fc_layers(x)
    
# ==============================================
class Sampler:

    def __init__(self, model, img_shape, sample_size, max_len=8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]

    def sample_new_exmps(self, steps=60, step_size=1):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)

        # Perform MCMC sampling
        inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)

        # Add new images to the buffer and remove old ones if needed
        self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(model, inp_imgs, steps=60, step_size=1, return_img_per_step=False):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.01)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs
        
# ===========================================
class DeepEnergyModel(pl.LightningModule):

    def __init__(self, img_shape, batch_size, alpha=0.1, lr=1e-4, beta1=0.0, **CNN_args):
        super().__init__()
        self.save_hyperparameters()

        # self.nn = CNNModel(**CNN_args)
        self.cnn = PointModel(**CNN_args)

        self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
        self.example_input_array = torch.zeros(1, *img_shape)

    def forward(self, x):
        z = self.cnn(x)
        return z

    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97) # Exponential decay over epochs
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        real_imgs, _ = batch
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples
        fake_imgs = self.sampler.sample_new_exmps(steps=60, step_size=10)

        # Predict energy score for all images
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        # Calculate losses
        reg_loss = self.hparams.alpha * (real_out ** 2 + fake_out ** 2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()
        loss = reg_loss + cdiv_loss

        # Logging
        self.log('loss', loss)
        self.log('loss_regularization', reg_loss)
        self.log('loss_contrastive_divergence', cdiv_loss)
        self.log('metrics_avg_real', real_out.mean())
        self.log('metrics_avg_fake', fake_out.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        # For validating, we calculate the contrastive divergence between purely random images and unseen examples
        # Note that the validation/test step of energy-based models depends on what we are interested in the model
        real_imgs, _ = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log('val_contrastive_divergence', cdiv)
        self.log('val_fake_out', fake_out.mean())
        self.log('val_real_out', real_out.mean())

# =======================================

class GenerateCallback(pl.Callback):

    def __init__(self, batch_size=8, vis_steps=5, num_steps=50, every_n_epochs=1):
        super().__init__()
        self.batch_size = batch_size         # Number of images to generate
        self.vis_steps = vis_steps           # Number of steps within generation to visualize
        self.num_steps = num_steps           # Number of steps to take during generation
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        # Skip for all other epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Generate images
            imgs_per_step = self.generate_imgs(pl_module)
            # Plot and add to tensorboard
            for i in range(imgs_per_step.shape[1]):
                step_size = self.num_steps // self.vis_steps
                imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
                grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1,1))
                trainer.logger.experiment.add_image(f"generation_{i}", grid, global_step=trainer.current_epoch)

    def generate_imgs(self, pl_module):
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
        imgs_per_step = Sampler.generate_samples(pl_module.cnn, start_imgs, steps=self.num_steps, step_size=10, return_img_per_step=True)
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step
    
# ===========================================
class SamplerCallback(pl.Callback):

    def __init__(self, num_imgs=32, every_n_epochs=1):
        super().__init__()
        self.num_imgs = num_imgs             # Number of images to plot
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            exmp_imgs = torch.cat(random.choices(pl_module.sampler.examples, k=self.num_imgs), dim=0)
            grid = torchvision.utils.make_grid(exmp_imgs, nrow=4, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("sampler", grid, global_step=trainer.current_epoch)

# =========================================
# =========================================
def train_model(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=100,
                         gradient_clip_val=0.1,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                                    GenerateCallback(every_n_epochs=5),
                                    SamplerCallback(every_n_epochs=5),
                                    LearningRateMonitor("epoch")
                                   ])

    pl.seed_everything(42)
    model = DeepEnergyModel(**kwargs)
    trainer.fit(model, train_loader, val_loader)
    # model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    return model

if __name__ == '__main__':
    # Rest of your code
    model = train_model(img_shape=(2,),
                        batch_size=train_loader.batch_size,
                        lr=1e-4,
                        beta1=0.0)
    # Rest of your code

# ============================================

    # pdb.set_trace()
    model.to(device)
    pl.seed_everything(43)
    vis_step = 2
    callback = GenerateCallback(batch_size=500, vis_steps=vis_step, num_steps=50)
    imgs_per_step = callback.generate_imgs(model)
    imgs_per_step = imgs_per_step.cpu()

    # Assuming you have a validation dataset with 2D points
    real_data, _ = next(iter(val_loader))  # Replace with your validation dataset

    # pdb.set_trace()
    # Plotting
    result_loc = "./../spiral_results/generated_images"
    os.makedirs(result_loc, exist_ok=True)
    
    gif_images = []
    for i in range(imgs_per_step.shape[0]//vis_step):
        index = i * vis_step
        imgs_to_plot = imgs_per_step[index]
        # imgs_to_plot = torch.cat([imgs_per_step[0:1, i], imgs_to_plot], dim=0)
        
        # pdb.set_trace()

        plt.scatter(real_data[:, 0], real_data[:, 1], color='blue', label='Real Data', alpha=0.5)
        plt.scatter(imgs_to_plot[:, 0], imgs_to_plot[:, 1], color='red', label='Generated Data', alpha=0.5)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.title(f"Step Size: {index+1}")
        plt.savefig(f'{result_loc}/generated_{i+1}.png', bbox_inches='tight')
        plt.close()
        
        img_path = f'{result_loc}/generated_{i + 1}.png'
        gif_images.append(Image.open(img_path))
    # Create a GIF        

    # Save the GIF
    gif_path = f'{result_loc}/GIF_GENERATED.gif'
    gif_images[0].save(
        gif_path,
        save_all=True,
        append_images=gif_images[1:],
        duration=500,  # Duration between frames in milliseconds
        loop=1,  # 0 for an infinite loop, other values for a finite loop
    )
    print(f"GIF saved at: {gif_path}")