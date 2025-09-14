import os
import numpy as np
import torch
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt

def load_vae(model_id="stabilityai/sd-vae-ft-ema", device=None):
    vae = AutoencoderKL.from_pretrained(model_id)
    torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(torch_device)
    vae.eval()
    return vae, torch_device

@torch.no_grad()
def decode_latents(vae, torch_device, latent_batch, std_latent=1.0, batch_size=32):
    tensor = torch.from_numpy((latent_batch * std_latent)).permute(0, 3, 1, 2).to(torch_device)
    decoded = []
    for i in range(0, tensor.shape[0], batch_size):
        chunk = tensor[i:i + batch_size]
        out = vae.decode(chunk).sample
        decoded.append(out.permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(decoded, axis=0)

def _imshow(ax, img):
    img = np.clip(img, -1, 1)
    img = (img + 1) / 2.0
    ax.imshow(img)
    ax.axis("off")

def plot_grid(imgs, save_path, ncols=None):
    n = len(imgs)
    ncols = ncols or int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    axs = np.array(axs).reshape(nrows, ncols)
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r, c]
            if idx < n:
                _imshow(ax, imgs[idx])
                idx += 1
            else:
                ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
