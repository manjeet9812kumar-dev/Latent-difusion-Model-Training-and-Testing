import os
import argparse
import numpy as np
from tensorflow import keras
from tqdm import tqdm

from ldm import get_network, Diffuser, add_noise, load_vae, decode_latents, plot_grid

def batch_generator(train_data, text_embeds, batch_size, label_dropout_p=0.15):
    idx = np.arange(len(train_data))
    while True:
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            ids = idx[i : i + batch_size]
            imgs = train_data[ids].copy()
            labels = text_embeds[ids].copy()
            mask = np.random.binomial(1, label_dropout_p, size=len(ids)).astype(bool)
            labels[mask] = 0.0
            noisy, noise_levels = add_noise(imgs, std=1.0)
            noise_levels = noise_levels[:, None, None, None]
            yield (noisy, noise_levels, labels), imgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_path", required=True)
    ap.add_argument("--text_path", required=True)
    ap.add_argument("--extra_latent_path", default=None)
    ap.add_argument("--extra_text_path", default=None)
    ap.add_argument("--latent_size", type=int, default=16)
    ap.add_argument("--latent_channels", type=int, default=4)
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--class_guidance", type=float, default=4.0)
    ap.add_argument("--diffusion_steps", type=int, default=70)
    ap.add_argument("--model_dir", default="runs/text_to_image")
    ap.add_argument("--preview_every", type=int, default=1)
    ap.add_argument("--preview_count", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "text_to_image.h5")

    latents = np.load(args.latent_path)
    texts = np.load(args.text_path)
    if args.extra_latent_path and args.extra_text_path:
        extra_lat = np.load(args.extra_latent_path)
        extra_txt = np.load(args.extra_text_path)
        latents = np.concatenate([latents, extra_lat], axis=0)
        texts = np.concatenate([texts, extra_txt], axis=0)

    std_latent = float(np.std(latents) * 2.5)
    latents = np.clip(latents, -std_latent, std_latent) / std_latent

    unet = get_network(args.latent_size, block_depth=3, emb_size=args.embed_dim, latent_channels=args.latent_channels)
    unet.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss="mae")

    diffuser = Diffuser(unet, class_guidance=args.class_guidance, diffusion_steps=args.diffusion_steps)
    vae, torch_device = load_vae()

    num_imgs = args.preview_count
    seeds = np.random.normal(0, 1, (num_imgs, args.latent_size, args.latent_size, args.latent_channels))
    base_idx = 10020
    labels = np.vstack([texts[base_idx + i] for i in range(5) for _ in range(5)])

    gen = batch_generator(latents, texts, batch_size=args.batch_size)
    steps_per_epoch = len(latents) // args.batch_size

    for epoch in range(args.epochs):
        unet.fit(gen, steps_per_epoch=steps_per_epoch, epochs=1, verbose=1)
        if (epoch + 1) % args.preview_every == 0:
            unet.save(model_path, save_format="tf")
            imgs_lat = diffuser.reverse_diffusion(seeds, labels)
            imgs = decode_latents(vae, torch_device, imgs_lat, std_latent=std_latent)
            out_png = os.path.join(args.model_dir, f"preview_epoch_{epoch+1:03d}.png")
            plot_grid(imgs, out_png, ncols=5)

if __name__ == "__main__":
    main()
