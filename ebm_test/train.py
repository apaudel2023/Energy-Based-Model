import numpy as np
import argparse
from tqdm import tqdm
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import load_model
from models import FCNet, ConvNet
from langevin import sample_langevin
from data import sample_2d_data

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=('8gaussians', '2spirals', 'checkerboard', 'rings', 'MNIST'))
parser.add_argument('model', choices=('FCNet', 'ConvNet'))
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate. default: 1e-3')
parser.add_argument('--stepsize', type=float, default=0.1, help='Langevin dynamics step size. default 0.1')
parser.add_argument('--n_steps', type=int, default=100, help='The number of Langevin dynamics steps. default 100')
parser.add_argument('--n_epoch', type=int, default=100, help='The number of training epochs. default 100')
parser.add_argument('--alpha', type=float, default=1., help='Regularizer coefficient. default 100')
args = parser.parse_args()

# load dataset
N_train = 5000
N_val = 1000
N_test = 5000

X_train = sample_2d_data(args.dataset, N_train)
X_val = sample_2d_data(args.dataset, N_val)
X_test = sample_2d_data(args.dataset, N_test)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
val_dataset = tf.data.Dataset.from_tensor_slices(X_val)
test_dataset = tf.data.Dataset.from_tensor_slices(X_test)

# Configure the datasets
batch_size = 32
num_workers = 8

# Shuffle and batch the training dataset
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(num_workers)
val_dataset = val_dataset.shuffle(buffer_size=len(X_val)).batch(batch_size).prefetch(num_workers)
test_dataset = test_dataset.shuffle(buffer_size=len(X_test)).batch(batch_size).prefetch(num_workers)

# Create iterators for the datasets
train_dl = iter(train_dataset)
val_dl = iter(val_dataset)
test_dl = iter(test_dataset)

# build model
if args.model == 'FCNet':
    model = FCNet(in_dim=2, out_dim=1, l_hidden=(200, 100, 50), activation='relu', out_activation='linear')
    model.summary()
elif args.model == 'ConvNet':
    model = ConvNet(in_chan=1, out_chan=1)

optimizer = Adam(lr=args.lr, clipvalue=0.1)  # Use clipvalue to clip gradients

model.compile(optimizer=optimizer, loss='mean_squared_error')

# train loop
for i_epoch in range(args.n_epoch):
    l_loss = []
    for pos_x in tqdm(train_dl):
        # pos_x = pos_x.cuda()

        neg_x = np.random.randn(*pos_x.shape)
        neg_x = sample_langevin(neg_x, model, args.stepsize, args.n_steps, intermediate_samples=False)

        pos_out = model.predict(pos_x)
        neg_out = model.predict(neg_x)

        loss = (pos_out - neg_out) + args.alpha * (pos_out**2 + neg_out**2)
        loss = np.mean(loss)
        l_loss.append(loss)

        model.train_on_batch(pos_x, np.zeros_like(pos_x))  # Use train_on_batch for manual control over updates

    print(np.mean(l_loss))
