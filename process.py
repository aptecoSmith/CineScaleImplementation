# process.py

import os
import glob
from PIL import Image
from cinescale import CineScale  # Import the CineScale class from cinescale.py

class Process:
    IMAGE_FILE_TYPES = ('.png', '.jpg', '.jpeg')

    def __init__(self, model_path='model.ckpt', shot_scale_model_path='model_shotscale_967.h5'):
        self.cinescale = CineScale(model_path, shot_scale_model_path)

    def process_images_in_folder(self, folder_path, skip_existing=False):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Check if the file matches any of the image file types
                if any(file.endswith(ext) for ext in self.IMAGE_FILE_TYPES):
                    file_path = os.path.join(root, file)
                    filename = os.path.basename(file_path)
                    base_name, ext = os.path.splitext(filename)

                    # Check if the text files already exist
                    angle_file_exists = os.path.exists(os.path.join(root, f"{base_name}.angle.txt"))
                    level_file_exists = os.path.exists(os.path.join(root, f"{base_name}.level.txt"))
                    shot_file_exists = os.path.exists(os.path.join(root, f"{base_name}.shot.txt"))

                    # Skip processing if skip_existing is True and the text files exist
                    if skip_existing and (angle_file_exists or level_file_exists or shot_file_exists):
                        continue

                    try:
                        # Predict angle and level
                        angle, level = self.cinescale.predict_angle_and_level_from_image(file_path)

                        # Print and write angle and level predictions to text files
                        print(f"Image: {filename}, Angle: {angle}, Level: {level}")
                        with open(os.path.join(root, f"{base_name}.angle.txt"), 'w') as angle_file:
                            angle_file.write(f"Angle: {angle}\n")
                        with open(os.path.join(root, f"{base_name}.level.txt"), 'w') as level_file:
                            level_file.write(f"Level: {level}\n")

                        # Predict shot scale
                        shot_scale = self.cinescale.predict_shot_scale_from_image(file_path)

                        # Print and write shot scale prediction to text file
                        print(f"Image: {filename}, Shot Scale: {shot_scale}")
                        with open(os.path.join(root, f"{base_name}.shot.txt"), 'w') as shot_file:
                            shot_file.write(f"Shot Scale: {shot_scale}\n")

                    except Exception as e:
                        print(f"An error occurred while processing image '{filename}': {e}")

# Example usage:
# processor = Process()
# processor.process_images_in_folder('path/to/image/folder', show_plots=True)
