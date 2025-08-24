# Latent-difusion-Model-Training-and-Testing

# Building a Latent Diffusion Model from Scratch (PyTorch)  
*A walkthrough of the code, the math, and the moving parts (my understanding) so you can follow the whole process*

## Why latent diffusion?

Training a diffusion model directly on 128×128 RGB images is expensive. A **latent diffusion model (LDM)** compresses each image into a small latent (our case: **4×16×16**), learns to denoise *there*, and uses a pretrained VAE decoder to get back to pixels. That single idea gives us huge speed, smaller models, and fewer GPU usage.

This notebook is an implementation of that. A **from-scratch UNet** that learns to denoise 4×16×16 latents.

## What you’ll find inside

- **A pretrained VAE** (from `stabilityai/sd-vae-ft-ema`) to decode latents back to images.
- **UNet, stitched together by `get_network()`.
- **Clean helper functions** that mirror your the notebook:
  - `sinusoidal_embedding`, `ResidualBlock`, `SpatialAttention`, `CrossAttention`
  - `DownBlock`, `UpBlock`, `get_network`
  - `add_noise`, `dynamic_thresholding`, `Diffuser`
  - functions for decoding the latents from unet using vae decoder and showing the images `decode_latents`, `plot_images`

## The pipeline:

1) **Images → Latents**  
   The VAE used in notebook is pretrained model which is publicly available. Our model sees only **latents** `z ∈ ℝ^{4×16×16}`.

2) **Forward (noising) process**  
   For each latent `x0`, we create a noisy version `x_t` by mixing in Gaussian noise at a chosen noise level.

3) **UNet predicts the clean latent**  
   Given `(x_t, noise_level, text_embedding)`, the UNet predicts `x̂0` (our estimate of the original clean latent).

4) **Loss**  
   We train with **MAE** on latents: `|x̂0 − x0|`.

5) **Sampling (reverse diffusion)**  
   Start from random noise, step through decreasing noise levels, use the UNet’s guesses, and walk back to a clean latent—then decode with the VAE to get an image.

---## The math (gentle but concrete)

### 1) Forward (noising) process — what we feed the UNet

We follow your notebook’s noise recipe (`add_noise`) which builds a **noise level** $\alpha \in [0,1]$ by

- sampling $u \sim \mathcal{N}(0,1)$, truncating to $[0,3)$, scaling by $1/3$;  
- then mapping $\alpha = \sin(u)$.

Let $\gamma = \sqrt{1-\alpha^2}$ (the “signal level”). For a clean latent $\mathbf{x}_0$ and Gaussian noise $\boldsymbol{\epsilon}\!\sim\!\mathcal{N}(0,I)$:
$$
\boxed{\ \mathbf{x}_t = \gamma\,\mathbf{x}_0 + \alpha\,\boldsymbol{\epsilon}\ }.
$$
That’s implemented in **`add_noise(...)`**.

#### Derivation: DDPM-style Markov chain → direct formula $x_t$ from $x_0$

The classic forward process is a Markov chain
$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1})=\mathcal{N}\!\big(\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t \mathbf{I}\big),
$$
with small $\beta_t \in (0,1)$. Define $\alpha_t = 1-\beta_t$ and $\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s$. Unrolling the chain yields the **closed form**:
$$
q(\mathbf{x}_t \mid \mathbf{x}_0)=\mathcal{N}\!\big(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\ (1-\bar{\alpha}_t)\mathbf{I}\big),
$$
so sampling gives the familiar direct formula
$$
\boxed{\ \mathbf{x}_t=\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}\ },\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(0,I).
$$

> Our code uses a simpler **per-step convex mixture** parameterization with a scalar **noise level** $\alpha_t\!\in[0,1]$ (plus a signal coefficient). You can view $\alpha_t$ as a user-friendly proxy for $\sqrt{1-\bar{\alpha}_t}$.

---

### 2) Conditioning — how we tell the model “where we are”

- **Noise embedding** (`sinusoidal_embedding`) turns the scalar $\alpha$ into a 32-dim vector using exponentially spaced frequencies $f_k \in [1,1000]$:
  $$
  e(\alpha) = \big[\sin(2\pi f_k \alpha),\ \cos(2\pi f_k \alpha)\big]_{k=1}^{16}.
  $$
- **Text embedding** is a 512-dim vector (e.g., CLIP/other), projected to 256 dims by a linear layer.

We tile both across $H\times W$ and **concatenate** → a conditioning map with **32 + 256** channels. The UNet reads this map everywhere.

---

### 3) The UNet — how it thinks

The network is built by **`get_network(latent_image_size, block_depth, emb_size, latent_channels)`**. Inside, we reuse these blocks:

- **`ResidualBlock`**: two convs + skip.
- **`SpatialAttention`**: self-attention over image positions.  
  With $Q,K,V$ from the image, the output is $\mathrm{softmax}(K^\top Q)\,V$.
- **`CrossAttention`**: queries/values from the **text/cond map**, keys from the **image**.  
  This lets text steer the image features (your original design).

Architecture sketch:

- **Down path**: 128 → 256 → 512 channels (attention from stage 2 onward). Uses `DownBlock(...)`, which stacks residuals + (optional) attention and pools.
- **Bottleneck**: widen to $128\times 5$ channels; apply spatial and cross attention a few times.
- **Up path**: mirror of down with `UpBlock(...)` and skip connections.

All this wiring happens inside **`get_network(...)`**; the helper functions are defined above it just like in your original notebook.

---

### 4) Objective — what we optimize

We train the model to estimate the clean latent $\mathbf{x}_0$:
$$
\boxed{\ \mathcal{L} = \big\| f_\theta(\mathbf{x}_t,\alpha,y) - \mathbf{x}_0 \big\|_1\ }.
$$
That’s “x₀-prediction” with **MAE** on latents.

---

### 5) Classifier-Free Guidance — how we push toward the text

We do **label dropout (15%)** during training so the model learns both conditional and unconditional behavior. At sampling, we run **two passes** and blend:
$$
\boxed{\ \hat{\mathbf{x}}_0 = w\,\hat{\mathbf{x}}_0^{(y)} + (1-w)\,\hat{\mathbf{x}}_0^{(\varnothing)}\ },
$$
with guidance weight $w$ (we use `class_guidance = 4.0`). That’s in **`Diffuser.predict_x_zero(...)`**.

---

### 6) Reverse diffusion — how we actually sample

We use a 70-step noise schedule (the same shape as your notebook). For step $i$ with current noise level $\alpha_i$ and next $\alpha_{i+1}$, we update latents via your **weighted blend**.

There are two equivalent ways to see the algebra—below is the **exact derivation used in code** (convex-mixture forward), and then the **general form** if you keep a separate signal coefficient.

#### (A) Derivation used in code (convex-mixture forward)

Assume the forward relation at step $i$ is the convex mix
$$
\mathbf{x}_i = (1-\alpha_i)\,\mathbf{x}_0 + \alpha_i\,\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(0,I).
$$
Given $\mathbf{x}_i$ and a model estimate $\hat{\mathbf{x}}_0$, solve for the implied noise:
$$
\boldsymbol{\epsilon}\ \approx\ \frac{\mathbf{x}_i - (1-\alpha_i)\,\hat{\mathbf{x}}_0}{\alpha_i}.
$$
To move to the next (smaller-noise) level $\alpha_{i+1}$, form the target convex mix:
$$
\mathbf{x}_{i+1} \ \approx\ (1-\alpha_{i+1})\,\hat{\mathbf{x}}_0 + \alpha_{i+1}\,\boldsymbol{\epsilon}.
$$
Substitute and simplify:
$$
\begin{aligned}
\mathbf{x}_{i+1}
&\approx (1-\alpha_{i+1})\,\hat{\mathbf{x}}_0
+ \alpha_{i+1}\,\frac{\mathbf{x}_i - (1-\alpha_i)\,\hat{\mathbf{x}}_0}{\alpha_i} \\
&= \frac{\alpha_{i+1}}{\alpha_i}\,\mathbf{x}_i
+ \Big[(1-\alpha_{i+1}) - \frac{\alpha_{i+1}}{\alpha_i}(1-\alpha_i)\Big]\,\hat{\mathbf{x}}_0 \\
&= \frac{\alpha_{i+1}}{\alpha_i}\,\mathbf{x}_i
+ \Big[1-\frac{\alpha_{i+1}}{\alpha_i}\Big]\,\hat{\mathbf{x}}_0 \\
&= \boxed{\ \frac{(\alpha_i-\alpha_{i+1})\,\hat{\mathbf{x}}_0 + \alpha_{i+1}\,\mathbf{x}_i}{\alpha_i}\ }.
\end{aligned}
$$
This is exactly the update implemented in **`Diffuser.reverse_diffusion(...)`**.

#### (B) General derivation (separate signal/noise coefficients)

If you prefer the normalized-energy form
$$
\mathbf{x}_i = \gamma_i\,\mathbf{x}_0 + \alpha_i\,\boldsymbol{\epsilon},\qquad \gamma_i=\sqrt{1-\alpha_i^2},
$$
the same eliminate-and-substitute steps give
$$
\boxed{\ \mathbf{x}_{i+1}
= \frac{\alpha_{i+1}}{\alpha_i}\,\mathbf{x}_i
+ \Big(\gamma_{i+1} - \frac{\alpha_{i+1}\gamma_i}{\alpha_i}\Big)\,\hat{\mathbf{x}}_0\ }.
$$
Our code uses the convex-mixture variant for simplicity (and because it matches your original update exactly). Both views are consistent ways to schedule noise; if you switch parameterization, update the blend accordingly.

---

### 7) Dynamic thresholding — how we keep predictions sane

Per sample, clip $\hat{\mathbf{x}}_0$ to a high percentile of its absolute values (99.5–99.75) and rescale to $[-1,1]$. This prevents rare spikes from derailing images. Code is in **`dynamic_thresholding(...)`**.

---

## How the pieces map to the code

- **Data & Viz**
  - `decode_latents(latents, std_latent)` — VAE decode (latents → RGB)
  - `plot_images(...)`, `imshow(...)` — small plotting helpers
- **Math & Noise**
  - `sinusoidal_embedding(noise_levels, emb_dim=32)`
  - `add_noise(array_nhwc)`
  - `dynamic_thresholding(img, perc=99.5)`
- **Model Blocks**
  - `ResidualBlock(in_ch, out_ch)`
  - `SpatialAttention(ch)`
  - `CrossAttention(img_ch, txt_ch)`
  - `DownBlock(in_ch, out_ch, block_depth, cond_ch, use_self_attention=True)`
  - `UpBlock(in_ch, out_ch, block_depth, cond_ch, use_self_attention=True)`
  - `get_network(latent_image_size, block_depth, emb_size, latent_channels)`
- **Sampling**
  - `Diffuser(denoiser, class_guidance, diffusion_steps, ...)`
    - `predict_x_zero(x_t, labels, noise_level_scalar)`
    - `reverse_diffusion(seeds_nhwc, labels_np)`
    - `predict_x_zero(x_t, labels, noise_level_scalar)`  
    - `reverse_diffusion(seeds_nhwc, labels_np)`
