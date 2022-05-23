import torch
import numpy as np


def create_fashion_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_gpu = True if torch.cuda.is_available() else False
    fashion = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=use_gpu)
    fashion_noise, _ = fashion.buildNoiseData(1)
    @torch.inference_mode()
    def fashion_generator(latents):
        latents = [torch.from_numpy(latent).float().to(device) for latent in latents]
        latents = torch.stack(latents)
        out = fashion.test(latents)
        outs = []
        for out_i in out:
            out_i = ((out_i.permute(1,2,0) + 1) * 127.5).clamp(0,255).cpu().numpy()
            out_i = np.uint8(out_i)
            outs.append(out_i)
        return outs

    return {
        'name': 'Fashion',
        'generator': fashion_generator,
        'latent_dim': fashion_noise.shape[1],
        'fps': 15,
        'batch_size': 8,
        'strength': 0.6,
        'max_duration': 30,
        'use_peak': True
    }
