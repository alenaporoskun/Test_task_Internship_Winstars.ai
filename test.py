# Import necessary libraries for working with models, images, and data visualization
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Load a pre-trained model
    loaded_model = load_model('model/model.keras')

    # Folder with test images
    test_image_folder = 'test_v2'

    # List of test images to check
    test_imgs = ['00dc34840.jpg', '00c3db267.jpg', '00aa79c47.jpg', '00a3a9d72.jpg']

    # Upload and resize test images
    resized_test_images = [np.array(image.load_img(os.path.join(test_image_folder, img), target_size=(768, 768))) for img in test_imgs]

    # Convert to a format suitable for model input
    resized_test_images = np.array(resized_test_images) / 255.0  # Нормалізація значень до [0, 1]

    # Making predictions on test images
    predictions = loaded_model.predict(resized_test_images)

    # Path to the folder where you want to save the image
    save_path = 'result-image'
    os.makedirs(save_path, exist_ok=True)

    # Saving the results
    for i in range(len(test_imgs)):
        # Get a test image and its prediction
        test_image = resized_test_images[i]
        prediction = predictions[i]

        # Graph the test image and its predictions
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(test_image)
        plt.title('Test Image')

        plt.subplot(1, 2, 2)
        plt.imshow(prediction.squeeze(), cmap='gray')
        plt.title('Model Prediction')

        # Save the graph to a file
        file_name = f'result_{os.path.splitext(test_imgs[i])[0]}.png'
        file_path = os.path.join(save_path, file_name)
        
        plt.savefig(file_path, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()