# cinescale.py

import torch
from torchvision import transforms
from PIL import Image
from resnet_model import ResNet  # Import the ResNet class from resnet_model.py
import tensorflow as tf  # Import TensorFlow for handling .h5 models
import numpy as np

class CineScale:
    def __init__(self, angle_and_level_model_path='camera_level_and_angle.ckpt', shot_scale_model_path='model_shotscale_967.h5'):
        # Determine the device to load the model on
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.angle_and_level_model_path = angle_and_level_model_path
        self.angle_and_level_model = self.load_resnet_model(self.device)

        self.shot_scale_model_path = shot_scale_model_path
        self.shot_scale_model = self.load_h5_model()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,   0.456,   0.406], std=[0.229,   0.224,   0.225]),
        ])

    def load_resnet_model(self,device):

        # Load the ResNet model from the checkpoint file on the determined device
        model = ResNet(num_angle_classes=5, num_level_classes=6)
        checkpoint = torch.load(self.angle_and_level_model_path, map_location=self.device)

        # Extract the state dict from the checkpoint if it exists
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Update the model's state dict with the checkpoint's state dict, ignoring missing keys
        model.load_state_dict(state_dict, strict=False)

        model.eval()

        # Move the model to the device if it's not already there
        if device.type == 'cuda':
            model = model.to(device)

        return model

    def predict_angle_and_level_from_image(self, image_path):
        angle_classes = ['Overhead angle', 'High angle', 'Neutral angle', 'Low angle', 'Dutch angle']
        level_classes = ['Aerial level', 'Eye level', 'Shoulder level', 'Hip level', 'Knee level', 'Ground level']
        img = Image.open(image_path)
        img_tensor = self.transform(img).unsqueeze(0)

        # img_tensor to device
        if self.device.type == 'cuda':
            img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            angle_logits, level_logits = self.angle_and_level_model(img_tensor)
        angle_pred = torch.argmax(angle_logits, dim=1).item()
        level_pred = torch.argmax(level_logits, dim=1).item()

        print(f"angle class: {angle_pred}")
        print(f"level class: {level_pred}")

        # Map the predicted indices to their corresponding class labels
        angle_label = f"{angle_pred}_{angle_classes[angle_pred]}"
        level_label = f"{level_pred}_{level_classes[level_pred]}"

        return angle_label, level_label

    def load_h5_model(self):
        # Load the .h5 model using TensorFlow's Keras API
        model = tf.keras.models.load_model(self.shot_scale_model_path)
        return model

    def predict_shot_scale_from_image(self, image_path):
        # Define the class labels
        #id2cls = ['Close Shot', 'Medium Shot', 'Long Shot']
        # id2cls = ['Close Shot', 'Medium Shot', 'Long Shot','unknown','unknown','unknown','unknown','unknown','unknown','unknown']
        # categories from https://cinescale.github.io/shotscale/#get-the-model Dataset part
        id2cls = [
            'Extreme Close Up',
            'Close Up',
            'Medium Close Up',
            'Medium Shot',
            'Medium Long Shot',
            'Long Shot',
            'Extreme Long Shot',
            'Foreground Shot',
            'Insert Shot',
        ]
        # Open the image
        img = Image.open(image_path)

        # Resize the image to the target dimensions
        width_height_tuple = (224,  125)
        img_resized = img.resize(width_height_tuple, Image.NEAREST)

        # Create a new image with the target dimensions filled with zeros
        final_img = Image.new('RGB', width_height_tuple, (0,  0,  0))

        # Paste the resized image onto the new image
        final_img.paste(img_resized, ((width_height_tuple[0] - img_resized.size[0]) //  2,
                                      (width_height_tuple[1] - img_resized.size[1]) //  2))

        # Convert the final image to a NumPy array and stack it with its grayscale version
        final_img_np = np.array(final_img, dtype='float32') /  255.
        final_img_gray_np = np.array(final_img.convert('L'), dtype='float32') /  255.

        # Ensure both arrays have the same shape by expanding the grayscale array to match the color image
        final_img_gray_np = np.expand_dims(final_img_gray_np, axis=-1)  # Add an extra dimension for the grayscale channel
        final_img_gray_np = np.repeat(final_img_gray_np,  3, axis=-1)  # Duplicate the grayscale channel three times to match the color image

        # Stack the color and grayscale images along the channel axis
        image = np.stack([final_img_np, final_img_gray_np], axis=0)

        # Use the TensorFlow model to make predictions
        with tf.device('/cpu:0'):  # Ensure the model runs on CPU
            shot_scale_prediction = self.shot_scale_model.predict(image)

        # Get the predicted class label
        predicted_class_index = np.argmax(shot_scale_prediction)

        # Check if the predicted class index is within the valid range
        if predicted_class_index >= len(id2cls):
            raise IndexError(f"Predicted class index ({predicted_class_index}) is out of range for id2cls list.")

        predicted_class_label = f"{predicted_class_index}_{id2cls[predicted_class_index]}"
        return predicted_class_label



