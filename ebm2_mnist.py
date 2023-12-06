import os

import numpy as np
import matplotlib.pyplot as plt
import torch
# from sklearn.datasets import load_digits
# from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from torchvision.datasets import MNIST
from torchsummary import summary


import pdb


class MNISTDataset(Dataset):
    def __init__(self, mode='train', transforms=None):
        mnist_dataset = MNIST(root='./data', train=(mode == 'train'), download=True, transform=None)
        
        if mode == 'train':
            self.data = mnist_dataset.data[:1000].float() / 255.0  # Normalize to [0, 1]
            self.targets = mnist_dataset.targets[:1000]
        elif mode == 'val':
            self.data = mnist_dataset.data[1000:1200].float() / 255.0
            self.targets = mnist_dataset.targets[1000:1200]
        else:
            self.data = mnist_dataset.data[1200:1300].float() / 255.0
            self.targets = mnist_dataset.targets[1200:1300]
        
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_x = self.data[idx].view(-1)  # Flatten to a 1D tensor
        sample_y = self.targets[idx]
        
        if self.transforms:
            sample_x = self.transforms(sample_x)
        
        return sample_x, sample_y
    
class EBM(nn.Module):
    def __init__(self, energy_net, alpha, sigma, ld_steps, D):
        super(EBM, self).__init__()

        print('EBM by JT.')

        # the neural net used by the EBM
        self.energy_net = energy_net

        # the loss for classification
        self.nll = nn.NLLLoss(reduction='none')  # it requires log-softmax as input!!

        # hyperparams
        self.D = D
        self.sigma = sigma
        self.alpha = torch.FloatTensor([alpha])
        self.ld_steps = ld_steps

    def classify(self, x):
        f_xy = self.energy_net(x)
        # pdb.set_trace()
        y_pred = torch.softmax(f_xy, 1)
        return torch.argmax(y_pred, dim=1)

    def class_loss(self, f_xy, y):
        # - calculate logits (for classification)
        y_pred = torch.softmax(f_xy, 1)
        return self.nll(torch.log(y_pred), y)

    def gen_loss(self, x, f_xy):
        # - sample using Langevine dynamics
        x_sample = self.sample(x=None, batch_size=x.shape[0])
        # - calculate f(x_sample)[y]
        f_x_sample_y = self.energy_net(x_sample)

        return -(torch.logsumexp(f_xy, 1) - torch.logsumexp(f_x_sample_y, 1))

    def forward(self, x, y, reduction='avg'):
        # =====
        # forward pass through the network
        # - calculate f(x)[y]
        f_xy = self.energy_net(x)

        # =====
        # discriminative part
        # - calculate the discriminative loss: the cross-entropy
        L_clf = self.class_loss(f_xy, y)

        # =====
        # generative part
        # - calculate the generative loss: E(x) - E(x_sample)
        L_gen = self.gen_loss(x, f_xy)

        # =====
        # Final objective
        if reduction == 'sum':
            loss = (L_clf + L_gen).sum()
        else:
            loss = (L_clf + L_gen).mean()

        return loss

    def energy_gradient(self, x):
        self.energy_net.eval()

        # copy original data that doesn't require grads!
        x_i = torch.FloatTensor(x.data)
        x_i.requires_grad = True  # WE MUST ADD IT, otherwise autograd won't work

        # calculate the gradient
        x_i_grad = torch.autograd.grad(torch.logsumexp(self.energy_net(x_i), 1).sum(), [x_i], retain_graph=True)[0]

        self.energy_net.train()

        return x_i_grad

    def langevine_dynamics_step(self, x_old, alpha):
        # Calculate gradient wrt x_old
        grad_energy = self.energy_gradient(x_old)
        # Sample eta ~ Normal(0, alpha)
        epsilon = torch.randn_like(grad_energy) * self.sigma

        # New sample
        x_new = x_old + alpha * grad_energy + epsilon

        return x_new

    def sample(self, batch_size=64, x=None):
        # - 1) Sample from uniform
        x_sample = 2. * torch.rand([batch_size, self.D]) - 1.
        # Sample from Gaussian distribution
        # x_sample = torch.randn([batch_size, self.D])
        # - 2) run Langevine Dynamics
        for i in range(self.ld_steps):
            x_sample = self.langevine_dynamics_step(x_sample, alpha=self.alpha)

        return x_sample
    
def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model')

    model_best.eval()
    loss = 0.
    loss_error = 0.
    loss_gen = 0.
    N = 0.
    for indx_batch, (test_batch, test_targets) in enumerate(test_loader):
        # pdb.set_trace()
        # hybrid loss
        loss_t = model_best.forward(test_batch, test_targets, reduction='sum')
        loss = loss + loss_t.item()
        # classification error
        # pdb.set_trace()
        y_pred = model_best.classify(test_batch)

        e = 1.*(y_pred == test_targets)
        loss_error = loss_error + (1. - e).sum().item()
        # generative nll
        f_xy_test = model_best.energy_net(test_batch)
        loss_gen = loss_gen + model_best.gen_loss(test_batch, f_xy_test).sum()
        # the number of examples
        N = N + test_batch.shape[0]
    loss = loss / N
    loss_error = loss_error / N
    loss_gen = loss_gen / N

    if epoch is None:
        print(f'FINAL PERFORMANCE: nll={loss}, ce={loss_error}, gen_nll={loss_gen}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}, val ce={loss_error}, val gen_nll={loss_gen}')

    return loss, loss_error, loss_gen


def samples_real(name, test_loader):
    # REAL-------
    num_x = 4
    num_y = 4
    x, _ = next(iter(test_loader))
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (28, 28))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name+'_real_images.png', bbox_inches='tight')
    plt.close()


def samples_generated(name, data_loader, extra_name=''):
    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 4
    num_y = 4
    x = model_best.sample(num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (28, 28))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.png', bbox_inches='tight')
    plt.close()


def plot_curve(name, nll_val, file_name='_nll_val_curve.png', color='b-'):
    plt.plot(np.arange(len(nll_val)), nll_val, color, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + file_name, bbox_inches='tight')
    plt.close()

def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    nll_val = []
    gen_val = []
    error_val = []
    best_nll = 1000.
    patience = 0

    # pdb.set_trace()
    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, (batch, targets) in enumerate(training_loader):

            loss = model.forward(batch, targets)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_e, error_e, gen_e = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_e)  # save for plotting
        gen_val.append(gen_e)  # save for plotting
        error_val.append(error_e)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, name + '.model')
            best_nll = loss_e
        else:
            if loss_e < best_nll:
                print('saved!')
                torch.save(model, name + '.model')
                best_nll = loss_e
                patience = 0

                samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)
    error_val = np.asarray(error_val)
    # gen_val = np.asarray(gen_val)
    # Assuming gen_val is a list of PyTorch tensors
    gen_val = [tensor.detach().cpu().numpy() for tensor in gen_val]
    # Now, gen_val_numpy is a list of NumPy arrays
    return nll_val, error_val, gen_val

# Define a custom transform
class MNISTCustomTransform:
    def __init__(self, add_noise=True):
        self.add_noise = add_noise

    def __call__(self, x):
        # Normalize to the range [-1, 1] # Convert to PyTorch tensor # Add random noise during training

        x = 2. * (x / 255.) - 1.
        if self.add_noise:
            x = x + 0.03 * torch.randn_like(x)
        
        return x
    
# Create a Compose transform with multiple operations for training
transforms_train = tt.Compose([
    MNISTCustomTransform(add_noise=True)
    ])

# Create a Compose transform with multiple operations for validation (no noise)
transforms_val = tt.Compose([
    MNISTCustomTransform(add_noise=False)
    ])

train_data = MNISTDataset(mode='train', transforms=transforms_train)
val_data = MNISTDataset(mode='val', transforms=transforms_val)
test_data = MNISTDataset(mode='test', transforms=transforms_val)

# pdb.set_trace()
training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# pdb.set_trace()

D = 28 * 28  # Input dimension for MNIST images
K = 10  # Output dimension for 10 classes
# M = 512  # The number of neurons

sigma = 0.05  # The noise level
alpha = 1000  # The step-size for SGLD
ld_steps = 60  # The number of steps of SGLD

lr = 1e-3  # Learning rate
num_epochs = 200  # Max. number of epochs
max_patience = 10  # Early stopping patience


name = 'ebm' + '_' + str(alpha) + '_' + str(sigma) + '_' + str(ld_steps)
result_dir = './../results/' + name + '/'
if not (os.path.exists(result_dir)):
    os.makedirs(result_dir, exist_ok=True)



class EnergyNet(nn.Module):
    def __init__(self, input_channels, output_classes):
        super(EnergyNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 14 * 14, 512)  # Adjust the input size based on your actual input dimensions
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()

        # Output layer
        self.fc_out = nn.Linear(256, output_classes)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc_out(x)
        return x



# energy_net = nn.Sequential(nn.Linear(D, M), nn.ReLU(),
#                                nn.Linear(M, M), nn.ReLU(),
#                                nn.Linear(M, M), nn.ReLU(),
#                                nn.Linear(M, K))

# Example usage
input_channels = 1  # For grayscale images, set to 1. For RGB images, set to 3.
output_classes = 10  # Assuming 10 classes for MNIST
energy_net = EnergyNet(input_channels, output_classes)

# We initialize the full model
model = EBM(energy_net, alpha=alpha, sigma=sigma, ld_steps=ld_steps, D=D)

# Print the model summary
# summary(model, (D,))

# pdb.set_trace()
# OPTIMIZER
optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)

nll_val, error_val, gen_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs,
                                       model=model, optimizer=optimizer,
                                       training_loader=training_loader, val_loader=val_loader)



test_loss, test_error, test_gen = evaluation(name=result_dir + name, test_loader=test_loader)
f = open(result_dir + name + '_test_loss.txt', "w")
f.write('NLL: ' + str(test_loss) + '\nCA: ' + str(test_error) + '\nGEN NLL: ' + str(test_gen))
f.close()

samples_real(result_dir + name, test_loader)
samples_generated(result_dir + name, test_loader)

plot_curve(result_dir + name, nll_val)
plot_curve(result_dir + name, error_val, file_name='_ca_val_curve.png', color='r-')
plot_curve(result_dir + name, gen_val, file_name='_gen_val_curve.png', color='g-')