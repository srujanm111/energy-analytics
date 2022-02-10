#!/usr/bin/env zsh

# get directory script is located in
DIR="$(cd "$(dirname "$0")" && pwd)"

cd $DIR/mask-the-face
read "IMG?Enter file path for image: "
python mask_the_face.py --path "$IMG" --mask_type 'surgical' --verbose --write_original_image
cd $DIR
