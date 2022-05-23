import torch
import numpy as np


def create_celeba256_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_gpu = True if torch.cuda.is_available() else False
    celeba256 = torch.hub.load(
        'facebookresearch/pytorch_GAN_zoo:hub', 
        'PGAN', 
        model_name='celebAHQ-256', 
        pretrained=True, 
        useGPU=use_gpu
    )
    celeba256_noise, _ = celeba256.buildNoiseData(1)
    @torch.inference_mode()
    def celeba256_generator(latents):
        latents = [torch.from_numpy(latent).float().to(device) for latent in latents]
        latents = torch.stack(latents)
        out = celeba256.test(latents)
        outs = []
        for out_i in out:
            out_i = ((out_i.permute(1,2,0) + 1) * 127.5).clamp(0,255).cpu().numpy()
            out_i = np.uint8(out_i)
            outs.append(out_i)
        return outs

    return {
        'name': 'Celeba256',
        'generator': celeba256_generator,
        'latent_dim': celeba256_noise.shape[1],
        'fps': 5,
        'batch_size': 1,
        'strength': 0.6,
        'max_duration': 20,
        'use_peak': True
    }
