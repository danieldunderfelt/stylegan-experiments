# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import argparse
import hashlib
import io
import os
import random
import string
import pickle
import numpy as np
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
import config

def main(seed = '', model_name = 'faces', output = config.result_dir):
    # Initialize TensorFlow.
    tflib.init_tf()

    model_options = {
        'faces': config.model_faces,
        'celebs': config.model_celebs,
        'cats': config.model_cats,
    }

    model = model_options[model_name]

    print('Using model: ' + model_name)

    if seed and len(seed) > 0:
        str = seed 
    else:
        str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    print('Using seed: ' + str)

    seed = int(hashlib.sha256(str.encode('utf-8')).hexdigest(), 16) % 10**5

    print(seed)

    # Load pre-trained network.
    with open(model, 'rb') as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    rnd = np.random.RandomState(seed)
    latents = rnd.randn(1, Gs.input_shape[1])

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    # Save image.
    if output and len(output) > 0:
        result_dir = output 
    else:
        result_dir = config.result_dir

    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(result_dir, str + '_' + model_name + '_' + 'generated.png')
    Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', default='', metavar='Seed', nargs='?', help='Set your seed string. It will use a random seed if unset.')
    parser.add_argument('-o', '--output', default='', help='The output directory where you want the file.')
    parser.add_argument('-m', '--model', default='faces', choices=['faces', 'celebs', 'cats'], help='The model that should be used to generate the image.')
    args = parser.parse_args()

    main(seed=args.seed, model_name=args.model, output=args.output)