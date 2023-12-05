
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms


class LoadData:

    def __init__(self):
        self.path = './data'
        # Transformations applied on each image => make them a tensor and normalize between -1 and 1
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                    ])

        # Loading the training dataset. We need to split it into a training and validation part
        self.train_set = MNIST(root=self.path, train=True, transform=transform, download=True)

        # Loading the test set
        self.test_set = MNIST(root=self.path, train=False, transform=transform, download=True)
    def load_data(self):
        # We define a set of data loaders that we can use for various purposes later.
        # Note that for actually training a model, we will use different data loaders
        # with a lower batch size.
        train_loader = DataLoader(self.train_set, batch_size=128, shuffle=True,  drop_last=True,  num_workers=4, pin_memory=True)
        test_loader  = DataLoader(self.test_set,  batch_size=256, shuffle=False, drop_last=False, num_workers=4)