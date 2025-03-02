import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to load and preprocess the images
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# Function to deprocess the image for visualization
def deprocess_image(tensor):
    tensor = tensor.numpy()
    tensor = tensor.squeeze()
    tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], 3))
    tensor = tensor + [103.939, 116.779, 123.68]
    tensor = np.clip(tensor, 0, 255).astype('uint8')
    return tensor

# Load images
content_image_path = 'C:/Users/kanna/Downloads/blue-moon-lake.jpg'  # Replace with your content image path
style_image_path = 'C:/Users/kanna/Downloads/starry_night.jpg'      # Replace with your style image path

content_image = load_and_preprocess_image(content_image_path)
style_image = load_and_preprocess_image(style_image_path)

# Extract features using VGG19
model = vgg19.VGG19(weights='imagenet', include_top=False)
for layer in model.layers:
    layer.trainable = False

# Select layers for content and style
content_layer = 'block5_conv2'  # Layer used for content
style_layers = [
    'block1_conv1', 
    'block2_conv1', 
    'block3_conv1', 
    'block4_conv1', 
    'block5_conv1'
]

# Function to get the output of specific layers
def get_model_outputs(model, style_layers, content_layer):
    outputs = [model.get_layer(layer).output for layer in style_layers]
    outputs += [model.get_layer(content_layer).output]
    return tf.keras.Model(inputs=model.input, outputs=outputs)

model = get_model_outputs(model, style_layers, content_layer)

# Function to compute the content loss
def content_loss(content, target):
    return K.sum(K.square(content - target))

# Function to compute the style loss
def style_loss(style, target):
    return K.sum(K.square(gram_matrix(style) - gram_matrix(target)))

# Function to compute the Gram matrix
def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    a = K.reshape(tensor, (-1, channels))
    gram = K.dot(K.transpose(a), a)
    return gram

# Function for total loss
def total_loss(outputs, content_weight=1e3, style_weight=1e-2):
    style_loss_value = K.sum([style_loss(output[0], output[1]) for output in zip(outputs[:-1], style_targets)])
    content_loss_value = content_loss(outputs[-1], content_target)
    return style_weight * style_loss_value + content_weight * content_loss_value

# Prepare the content and style targets
content_target = model(content_image)[-1]
style_targets = [model(style_image)[i] for i in range(len(style_layers))]

# Create a new image as target (the initial generated image)
generated_image = tf.Variable(tf.keras.initializers.RandomNormal()(shape=content_image.shape), trainable=True)

# Optimize
optimizer = tf.keras.optimizers.Adam(learning_rate=5.0)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        outputs = model(generated_image)
        loss = total_loss(outputs)
    gradients = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(gradients, generated_image)])

# Run the optimization
iterations = 100
for i in range(iterations):
    train_step()
    if i % 10 == 0:
        img = deprocess_image(generated_image)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Iteration {i}')
        plt.show()

# Final result
img = deprocess_image(generated_image)
plt.imshow(img)
plt.axis('off')
plt.title('Final Output')
plt.show()
