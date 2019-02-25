# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for reproducing the figures of the StyleGAN paper using pre-trained generators."""

import os
import pickle
import numpy as np
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
import config

#----------------------------------------------------------------------------
# Helpers for loading and using pre-trained generators.

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()

def load_Gs(model):
	if model not in _Gs_cache:
		with open(model, 'rb') as f:
			_G, _D, Gs = pickle.load(f)
		_Gs_cache[model] = Gs
	return _Gs_cache[model]

#----------------------------------------------------------------------------
# Figure 3: Style mixing.

def draw_style_mixing_figure(png, Gs, w, h, src_seeds, dst_seeds, style_ranges):
	print(png)

	src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
	dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
	src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
	dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]
	src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
	dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)

	canvas = Image.new('RGB', (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), 'white')
	for col, src_image in enumerate(list(src_images)):
		canvas.paste(Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
	for row, dst_image in enumerate(list(dst_images)):
		print(row)
		canvas.paste(Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
		row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
		row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
		row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
		for col, image in enumerate(list(row_images)):
			canvas.paste(Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
	canvas.save(png)

#----------------------------------------------------------------------------
# Main program.

def main():
	tflib.init_tf()
	os.makedirs(config.result_dir, exist_ok=True)
	draw_style_mixing_figure(os.path.join(config.result_dir, 'figure03-style-mixing.png'), load_Gs(config.model_celebs), w=1024, h=1024, src_seeds=[639,123,687,615,2268,678], dst_seeds=[888,536,1898,1733,1614,251], style_ranges=[range(0,2)]+[range(2,4)]+[range(4,8)]+[range(8,10)]+[range(10,12)]+[range(12,14)])

#----------------------------------------------------------------------------

if __name__ == "__main__":
	main()

#----------------------------------------------------------------------------
