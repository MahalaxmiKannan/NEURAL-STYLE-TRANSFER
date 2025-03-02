# NEURAL-STYLE-TRANSFER
COMPANY: CODETECH IT SOLUTIONS

NAME: MAHALAXMI K

INTERN ID: CT08ROW

COMPANY: AI

DURATION: 4 WEEKS

MENTOR: NEELA SANTHOSH KUMAR

DESCRIPTION OF THE PROJECT

This project implements a neural style transfer technique using TensorFlow and VGG19 to blend the content of one image with the artistic style of another. Neural style transfer is a deep learning method that enables the generation of a new image that retains the content of a reference image (e.g., a photograph) while adopting the visual style of another (e.g., a painting). The process begins by loading and preprocessing two images: a content image and a style image. Both images are resized to 224x224 pixels to match the input requirements of the VGG19 model, which is a pre-trained Convolutional Neural Network (CNN) typically used for image classification tasks. The VGG19 model is utilized to extract feature representations from multiple layers, specifically chosen for their ability to capture both content and style information. The content representation is extracted from the fifth convolutional layer of the model, while the style representation is derived from the first convolutional layers of the network, which capture fine-grained textures and patterns. The core of the style transfer algorithm is to minimize a loss function that combines two components: content loss and style loss. Content loss measures the difference between the content of the generated image and the content image, while style loss compares the style of the generated image with the style image using the Gram matrix, which encodes the correlations between different feature maps. These losses are weighted, with the content loss typically having a higher weight, and are optimized using the Adam optimizer to iteratively update a generated image. The generated image starts as a random noise image and gradually evolves to resemble the content of the content image while adopting the style of the style image. The optimization process involves computing gradients with respect to the loss function and applying those gradients to update the image in each iteration. Every few iterations, the updated image is displayed to visualize the progress of the transformation, with the final output showcasing a visually stunning blend of content and style. This project leverages TensorFlow’s eager execution and automatic differentiation features to perform the optimization, allowing for a flexible and efficient implementation. The end result is a stylized image that combines the distinctive elements of the chosen style with the underlying structure of the content image, demonstrating the power of deep learning techniques in the domain of computer vision and artistic creation.

INPUT IMAGE 1
![Dawn Sky](https://github.com/user-attachments/assets/218323dc-2976-4bed-9244-bdcbb2e220a2)

INUPUT IMAGE 2
![starry_night](https://github.com/user-attachments/assets/267b5084-b013-4cd3-a3ac-64901a4c669d)

OUTPUT IMAGE
![image](https://github.com/user-attachments/assets/a92efe6c-dc0b-4f33-ad5a-b292b5f3311b)
