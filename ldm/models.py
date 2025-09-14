import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _attention_lambda(qkv):
    q, k, v = qkv
    attn_logits = tf.matmul(k, q, transpose_b=True)
    score = tf.nn.softmax(attn_logits, axis=-1)
    o = tf.matmul(score, v)
    return o

def spatial_attention(x):
    filters = x.shape[-1]
    orig_shape = (x.shape[1], x.shape[2], x.shape[3])
    h = layers.BatchNormalization()(x)
    q = layers.Conv2D(filters // 8, 1, padding="same")(h)
    k = layers.Conv2D(filters // 8, 1, padding="same")(h)
    v = layers.Conv2D(filters, 1, padding="same")(h)
    q = layers.Reshape((q.shape[1] * q.shape[2], q.shape[3]))(q)
    k = layers.Reshape((k.shape[1] * k.shape[2], k.shape[3]))(k)
    v = layers.Reshape((v.shape[1] * v.shape[2], v.shape[3]))(v)
    o = layers.Lambda(_attention_lambda)([q, k, v])
    o = layers.Reshape(orig_shape)(o)
    o = layers.Conv2D(filters, 1, padding="same")(o)
    o = layers.BatchNormalization()(o)
    return o

def cross_attention(img, txt):
    filters = img.shape[-1]
    orig_shape = (img.shape[1], img.shape[2], img.shape[3])
    i = layers.BatchNormalization()(img)
    t = layers.BatchNormalization()(txt)
    q = layers.Conv2D(filters // 8, 1, padding="same")(t)
    k = layers.Conv2D(filters // 8, 1, padding="same")(i)
    v = layers.Conv2D(filters, 1, padding="same")(t)
    q = layers.Reshape((q.shape[1] * q.shape[2], q.shape[3]))(q)
    k = layers.Reshape((k.shape[1] * k.shape[2], k.shape[3]))(k)
    v = layers.Reshape((v.shape[1] * v.shape[2], v.shape[3]))(v)
    o = layers.Lambda(_attention_lambda)([q, k, v])
    o = layers.Reshape(orig_shape)(o)
    o = layers.Conv2D(filters, 1, padding="same")(o)
    o = layers.BatchNormalization()(o)
    return o

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    embedding_max_frequency = 1000.0
    embedding_dims = 32
    freqs = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    ang = 2.0 * math.pi * freqs
    emb = tf.concat([tf.sin(ang * x), tf.cos(ang * x)], axis=3)
    return emb

def ResidualBlock(ch):
    def apply(x):
        in_ch = x.shape[-1]
        residual = x if in_ch == ch else layers.Conv2D(ch, 1)(x)
        h = layers.BatchNormalization(center=False, scale=False)(x)
        h = layers.Conv2D(ch, 3, padding="same", activation=keras.activations.swish)(h)
        h = layers.Conv2D(ch, 3, padding="same")(h)
        return layers.Add()([h, residual])
    return apply

def DownBlock(ch, depth, use_attn=True):
    def apply(inp):
        x, skips, embtxt = inp
        for _ in range(depth):
            x = ResidualBlock(ch)(x)
            if use_attn:
                x = layers.Add()([x, spatial_attention(x)])
                x = layers.Add()([x, cross_attention(x, embtxt)])
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x
    return apply

def UpBlock(ch, depth, use_attn=True):
    def apply(inp):
        x, skips, embtxt = inp
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        for _ in range(depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(ch)(x)
            if use_attn:
                x = layers.Add()([x, spatial_attention(x)])
                x = layers.Add()([x, cross_attention(x, embtxt)])
        return x
    return apply

def get_network(latent_image_size, block_depth=3, emb_size=512, latent_channels=4):
    noisy_images = keras.Input(shape=(latent_image_size, latent_image_size, latent_channels))
    x = layers.Conv2D(128, 1)(noisy_images)
    noise_variances = keras.Input(shape=(1, 1, 1))
    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=(latent_image_size, latent_image_size), interpolation="nearest")(e)
    input_label = keras.Input(shape=(emb_size,))
    emb_label = layers.Dense(emb_size // 2)(input_label)
    emb_label = layers.Reshape((1, 1, emb_size // 2))(emb_label)
    emb_label = layers.UpSampling2D(size=(latent_image_size, latent_image_size), interpolation="nearest")(emb_label)
    emb_and_noise = layers.Concatenate()([e, emb_label])
    skips = []
    x = DownBlock(128, block_depth, use_attn=False)([x, skips, emb_and_noise])
    emb_and_noise = layers.AveragePooling2D(pool_size=(2, 2))(emb_and_noise)
    x = DownBlock(256, block_depth, use_attn=True)([x, skips, emb_and_noise])
    emb_and_noise = layers.AveragePooling2D(pool_size=(2, 2))(emb_and_noise)
    x = DownBlock(512, block_depth, use_attn=True)([x, skips, emb_and_noise])
    emb_and_noise = layers.AveragePooling2D(pool_size=(2, 2))(emb_and_noise)
    for _ in range(block_depth):
        x = ResidualBlock(128 * 5)(x)
        x = layers.Add()([x, spatial_attention(x)])
        x = layers.Add()([x, cross_attention(x, emb_and_noise)])
    x = UpBlock(512, block_depth, use_attn=True)([x, skips, emb_and_noise])
    emb_and_noise = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(emb_and_noise)
    x = UpBlock(256, block_depth, use_attn=True)([x, skips, emb_and_noise])
    emb_and_noise = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(emb_and_noise)
    x = UpBlock(128, block_depth, use_attn=True)([x, skips, emb_and_noise])
    emb_and_noise = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(emb_and_noise)
    out = layers.Conv2D(latent_channels, 1, kernel_initializer="zeros")(x)
    return keras.Model([noisy_images, noise_variances, input_label], out, name="unet")
