import os
import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    '''Refer to https://github.com/safwankdb/Vanilla-GAN'''
    def __init__(self):
        super(Generator, self).__init__()
        self.n_features = 128
        self.n_out = 784
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_features, 256),
                    nn.LeakyReLU(0.2)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.LeakyReLU(0.2)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, 784),
                    nn.Tanh()
                    )
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 1, 28, 28)
        return x


def create_mnist_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mnist = Generator()
    state = torch.load(
        os.path.join(
            os.path.dirname(__file__),
            'mnist/mnist_generator.pretrained'
        ), 
        map_location='cpu'
    )
    mnist.load_state_dict(state)
    mnist.to(device)
    mnist.eval()

    @torch.inference_mode()
    def mnist_generator(latents):
        latents = [torch.from_numpy(latent).float().to(device) for latent in latents]
        latents = torch.stack(latents)
        out = mnist(latents)
        outs = []
        for out_i in out:
            out_i = ((out_i[0] + 1) * 127.5).clamp(0,255).cpu().numpy()
            out_i = np.uint8(out_i)
            out_i = np.stack([out_i]*3, -1)
            outs.append(out_i)
        return outs
    
    return {
        'name': 'MNIST',
        'generator': mnist_generator,
        'latent_dim': 128,
        'fps': 20,
        'batch_size': 8,
        'strength': 0.75,
        'max_duration': 30,
        'use_peak': True
    }
