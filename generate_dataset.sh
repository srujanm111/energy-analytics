#!/usr/bin/env bash

# paths
dataset_dir='dataset'
processed_files="$dataset_dir/processed_files.txt"
masked_faces_dir="$dataset_dir/masked_faces"
segmentation_masks_dir="$dataset_dir/segmentation_masks"

# clear previously processed files for new run
rm -f "$processed_files"
touch "$processed_files"

# create output directories if needed
mkdir "$masked_faces_dir" "$segmentation_masks_dir"

# face mask configurations
mask_type_list=( 'surgical' 'KN95' 'cloth' )
mask_colors_list=( '#0473e2' '#000000' '#ffffff' '#1fc49b' )

# function that takes an image of an unmasked face as input,
# and outputs the masked face image and segmentation mask
generate_images () {
    input_img_path=$1
    img_name=$(basename "$input_img_path")
    masked_face_img_path="$masked_faces_dir/$img_name"
    segmentation_mask_img_path="$segmentation_masks_dir/$img_name"

    # TODO randomize mask type and color
    mask_type=${mask_type_list[$(( RANDOM % 3 + 1 ))]}
    color=${mask_colors_list[$(( RANDOM % 4 + 1 ))]}

    # generate masked face image, then generate segmentation mask
    # only if a a face was detected in the input image
    if python mask-the-face/mask_the_face.py\
      --input "../$input_img_path"\
      --output "../$masked_face_img_path"\
      --mask_type "$mask_type"\
      --color "$color"\
      --write_original_image > /dev/null
    then
        python segmentation_mask.py "$input_img_path" "$masked_face_img_path" "$segmentation_mask_img_path"
        echo "$img_name" >> "$processed_files"
    fi
}

# generate images from the celeba dataset
for FILE in dataset/celeba/*; do
    printf '\r%s' "$FILE"
    generate_images "$FILE"
done
