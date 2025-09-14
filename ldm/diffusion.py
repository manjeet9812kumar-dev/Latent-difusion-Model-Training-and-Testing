import numpy as np

def add_noise(array, std=1.0):
    x = np.abs(np.random.normal(0, std, 2 * len(array)))
    x = x[x < 3] / 3
    noise_levels = x[: len(array)]
    noise_levels = np.sin(noise_levels)
    signal_levels = np.sqrt(1 - np.square(noise_levels))
    nl = noise_levels[:, None, None, None]
    sl = signal_levels[:, None, None, None]
    pure_noise = np.random.normal(0, 1, size=array.shape).astype("float32")
    noisy = array * sl + pure_noise * nl
    return noisy, noise_levels

def dynamic_thresholding(img, perc=99.5):
    s = np.percentile(np.abs(img.ravel()), perc)
    s = max(s, 1.0)
    return img.clip(-s, s) / s

class Diffuser:
    def __init__(self, denoiser, class_guidance, diffusion_steps, perc_thresholding=99.75, batch_size=64):
        self.denoiser = denoiser
        self.class_guidance = class_guidance
        self.diffusion_steps = diffusion_steps
        self.noise_levels = 1 - np.power(np.arange(0.0001, 0.99, 1 / diffusion_steps), 1 / 3)
        self.noise_levels[-1] = 0.01
        self.perc_thresholding = perc_thresholding
        self.batch_size = batch_size

    def predict_x0(self, x_t, label, noise_level):
        n = len(x_t)
        empty = np.zeros_like(label)
        noise_in = np.array([noise_level] * n)[:, None, None, None]
        nn_inputs = [np.vstack([x_t, x_t]), np.vstack([noise_in, noise_in]), np.vstack([label, empty])]
        x0 = self.denoiser.predict(nn_inputs, batch_size=self.batch_size, verbose=0)
        x0_lab, x0_un = x0[:n], x0[n:]
        x0 = self.class_guidance * x0_lab + (1 - self.class_guidance) * x0_un
        return dynamic_thresholding(x0, self.perc_thresholding)

    def reverse_diffusion(self, seeds, label):
        xt = seeds.copy()
        for i in range(len(self.noise_levels) - 1):
            curr_n, next_n = self.noise_levels[i], self.noise_levels[i + 1]
            x0 = self.predict_x0(xt, label, curr_n)
            xt = ((curr_n - next_n) * x0 + next_n * xt) / curr_n
        return x0
