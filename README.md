### requirements
Add a models folder to your root, place two models there
expecting:
camera_level_and_angle.ckpt - renamed from original download, from https://cinescale.github.io/camera_al/#get-the-model
model_shotscale_967.h5 from https://cinescale.github.io/shotscale/#get-the-model

### App creates three files per image
.angle.txt
.level.txt
.shot.txt

example usage:

python main.py --folder_path "/absolute/path/to/image/folder" --angle_and_level_model_path "models/camera_level_and_angle.ckpt" --shot_scale_model_path "models/model_shotscale_967.h5"

### credits
all credit to https://cinescale.github.io/