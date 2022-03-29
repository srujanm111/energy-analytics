# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import argparse
from mask_the_face.utils.aux_functions import *
import dlib

# import this function to perform masking on a single image via python function call
def mask_the_face(input_img, mask_type, color):
    parser = argparse.ArgumentParser(
        description="MaskTheFace - Python code to mask faces dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to the image itself",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to output the masked image",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="surgical",
        choices=["surgical", "N95", "KN95", "cloth", "gas", "inpaint", "random", "all"],
        help="Type of the mask to be applied. Available options: all, surgical_blue, surgical_green, N95, cloth",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="",
        help="Type of the pattern. Available options in masks/textures",
    )
    parser.add_argument(
        "--pattern_weight",
        type=float,
        default=0.5,
        help="Weight of the pattern. Must be between 0 and 1",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="#0473e2",
        help="Hex color value that need to be overlayed to the mask",
    )
    parser.add_argument(
        "--color_weight",
        type=float,
        default=0.5,
        help="Weight of the color intensity. Must be between 0 and 1",
    )
    parser.add_argument(
        "--code",
        type=str,
        # default="cloth-masks/textures/check/check_4.jpg, cloth-#e54294, cloth-#ff0000, cloth, cloth-masks/textures/others/heart_1.png, cloth-masks/textures/fruits/pineapple.png, N95, surgical_blue, surgical_green",
        default="",
        help="Generate specific formats",
    )
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Turn verbosity on"
    )
    parser.add_argument(
        "--write_original_image",
        dest="write_original_image",
        action="store_true",
        help="If true, original image is also stored in the masked folder",
    )
    parser.set_defaults(feature=False)

    args = parser.parse_args(['--mask_type', mask_type, '--color', color])

    # Set up dlib face detector and predictor
    args.detector = dlib.get_frontal_face_detector()
    path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(path_to_dlib_model):
        download_dlib_model()

    args.predictor = dlib.shape_predictor(path_to_dlib_model)

    masked_image, mask, mask_binary_array, original_image = mask_image(
        input_img, args
    )
    if len(mask) == 1:  
        return True, masked_image[0]

    return False, None
    