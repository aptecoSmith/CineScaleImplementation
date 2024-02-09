import argparse
import tensorflow as tf
from process import Process

def main(args):
    # Instantiate the Process class with the model paths provided as arguments
    processor = Process(args.angle_and_level_model_path, args.shot_scale_model_path)

    print('models loaded')
    # Use the folder path provided as an argument
    folder_path = args.folder_path

    # Call the process_images_in_folder method
    processor.process_images_in_folder(folder_path, skip_existing=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images using specified models.')
    parser.add_argument('--folder_path', type=str, required=True, help='The absolute path to the folder containing the images.')
    parser.add_argument('--angle_and_level_model_path', type=str, required=True, help='The path to the angle and level model file.')
    parser.add_argument('--shot_scale_model_path', type=str, required=True, help='The path to the shot scale model file.')

    args = parser.parse_args()
    main(args)
