import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
print("Модель завантажена!")

def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не знайдено!")
        return

    original_image = tf.image.decode_image(tf.io.read_file(image_path))
    print(f"Розмір зображення: {original_image.shape}")

    if original_image.shape[-1] == 4:
        original_image = original_image[..., :-1]

    hr_size = (tf.convert_to_tensor(original_image.shape[:-1]) // 4) * 4
    original_image = tf.image.crop_to_bounding_box(original_image, 0, 0, hr_size[0], hr_size[1])
    original_image = tf.cast(original_image, tf.float32)
    original_image = tf.expand_dims(original_image, 0)

    print("Покрашення зображення...")
    enhanced_image = model(original_image)
    enhanced_image = tf.squeeze(enhanced_image)
    print("Зображення покращено!")

    plt.rcParams['figure.figsize'] = [15, 10]
    fig, axes = plt.subplots(1, 2)
    fig.tight_layout()

    plt.subplot(1, 2, 1)
    plot_image(tf.squeeze(original_image), title='Original Image')

    plt.subplot(1, 2, 2)
    plot_image(tf.squeeze(enhanced_image), title='Enhanced Image')

    output_path = "ENHANCED_" + os.path.basename(image_path)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Збережено: {output_path}")


def plot_image(image, title="plot"):
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)


process_image("0017x8.png")
