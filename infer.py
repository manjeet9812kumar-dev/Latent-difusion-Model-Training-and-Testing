import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import clip

from ldm import get_network, Diffuser, load_vae, decode_latents, plot_grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--num_images", type=int, default=4)
    ap.add_argument("--latent_size", type=int, default=16)
    ap.add_argument("--latent_channels", type=int, default=4)
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--class_guidance", type=float, default=4.0)
    ap.add_argument("--diffusion_steps", type=int, default=70)
    ap.add_argument("--std_latent", type=float, default=None)
    ap.add_argument("--out_dir", default="runs/samples")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    unet = get_network(args.latent_size, block_depth=3, emb_size=args.embed_dim, latent_channels=args.latent_channels)
    unet.load_weights(args.weights)
    diffuser = Diffuser(unet, class_guidance=args.class_guidance, diffusion_steps=args.diffusion_steps)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    tokens = clip.tokenize([args.prompt] * args.num_images, truncate=False).to(device)
    with torch.no_grad():
        txt = clip_model.encode_text(tokens)
    text_encoding = txt.detach().cpu().numpy()

    seeds = np.random.normal(0, 1, (args.num_images, args.latent_size, args.latent_size, args.latent_channels))
    imgs_lat = diffuser.reverse_diffusion(seeds, text_encoding)

    vae, torch_device = load_vae()
    std_lat = float(args.std_latent) if args.std_latent is not None else 1.0
    imgs = decode_latents(vae, torch_device, imgs_lat, std_latent=std_lat)

    out_png = os.path.join(args.out_dir, f"{args.prompt.replace(' ', '_')}.png")
    plot_grid(imgs, out_png, ncols=int(np.ceil(np.sqrt(args.num_images))))
    print(out_png)

if __name__ == "__main__":
    main()
