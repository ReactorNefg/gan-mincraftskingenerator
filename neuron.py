import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout
from tensorflow.keras.optimizers import AdamW


# Создаем директорию, если ее нет
if not os.path.exists('generated_images'):
    os.makedirs('generated_images')

# Задаем параметры
input_shape = (64, 64, 3)
latent_dim = 100
batch_size = 16
epochs = 20000
save_interval = 1000

# Загружаем изображения из папки
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        # Загружаем изображение в цветовом формате RGB и размере 64x64
        img = load_img(os.path.join(directory, filename), color_mode='rgb', target_size=input_shape[:2])
        # Преобразуем изображение в массив numpy
        img = img_to_array(img)
        # Нормализуем значения пикселей до интервала [-1, 1]
        img = (img - 127.5) / 127.5
        # Добавляем изображение в список
        images.append(img)
    # Преобразуем список изображений в массив numpy
    images = np.asarray(images)
    return images


# Создаем генератор
def create_generator():
    inputs = keras.Input(shape=(latent_dim,))
    x = Dense(8*8*256)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((8, 8, 256))(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Conv2D(3, kernel_size=5, activation='tanh', padding='same')(x)
    generator = keras.Model(inputs, outputs)
    return generator


# Создаем дискриминатор
def create_discriminator():
    input_shape = (64, 64, 3)
    inputs = keras.Input(shape=input_shape)
    x = Conv2D(64, kernel_size=5, strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    discriminator = keras.Model(inputs, outputs)
    discriminator.compile(loss='binary_crossentropy', optimizer=AdamW(learning_rate=0.0002, beta_1=0.5))

    return discriminator


def create_gan(discriminator, generator):
    discriminator.trainable = False
    inputs = keras.Input(shape=(latent_dim,))
    generated_images = generator(inputs)
    outputs = discriminator(generated_images)
    gan = keras.Model(inputs, outputs)
    gan.compile(loss='binary_crossentropy', optimizer=AdamW(learning_rate=0.0002, beta_1=0.5))
    return gan


def save_img(filename, image_array):
    image_array = np.squeeze(image_array, axis=0)
    
    img = tf.keras.preprocessing.image.array_to_img(image_array)
    img = img.convert('RGBA')
    img.save(os.path.join('generated_images', f'generated_{epoch+1}.png'))



directory = 'skins'
images = load_images(directory)

generator = create_generator()
discriminator = create_discriminator()
gan = create_gan(discriminator, generator)

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

for epoch in range(epochs):
    # Выбираем случайные изображения из набора данных
    real_images = images[np.random.randint(0, images.shape[0], size=batch_size)]
    # Генерируем случайный шум для входа генератора
    noise = generate_latent_points(latent_dim, batch_size)
    # Генерируем изображения с помощью генератора
    fake_images = generator.predict(noise)
    # Объединяем настоящие и сгенерированные изображения
    x = np.concatenate((real_images, fake_images))
    # Создаем метки для обучения дискриминатора
    y = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))
    # Добавляем случайный шум в метки
    y += 0.05 * np.random.random(y.shape)
    # Обучаем дискриминатор на настоящих и сгенерированных изображениях
    d_loss = discriminator.train_on_batch(x, y)
    # Генерируем новый шум для обучения генератора
    noise = generate_latent_points(latent_dim, batch_size)

    y = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, y)

    if (epoch+1) % 10 == 0:
        print(f'Epoch {(epoch+1)}, d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')
        # Генерируем изображение с помощью генератора и сохраняем его
        img_array = generator.predict(generate_latent_points(latent_dim, 1))
        
        save_img(f'generated_{epoch+1}.png', img_array)