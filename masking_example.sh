#!/usr/bin/env zsh

read "IMG?Enter file path for image: "
python mask-the-face/mask_the_face.py --input "$IMG" --color_weight='0' --mask_type 'random' --verbose --write_original_image
