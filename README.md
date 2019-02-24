# StyleGAN
### But running on my machine

StyleGAN is a Tensorflow implementation of a GAN (Generative Adversarial Network) by Tero Karras, Samuli Laine and Timo Aila who are researchers at Nvidia. StyleGAN is the mastermind behind [This Person Does Not Exist](https://thispersondoesnotexist.com/) and similar cool sites. The StyleGAN implementation is licensed under Creative Commons BY-NC 4.0 and it is available here:

[NVlabs/stylegan](https://github.com/NVlabs/stylegan)

Find the research paper here: [https://arxiv.org/abs/1812.04948](https://arxiv.org/abs/1812.04948)

I wanted to run StyleGAN on my own machine, and this repository contains the code that works for me. I don't intend this to be generally applicable to everyone, but if it helps you out then great.

I copied the `dnnlib` dependency and the "pretrained example" code from the repo above and modified the example app into a version that runs on my machine plus some conveniences:

- Loads the pretrained model from the local filesystem instead of from Google Drive
- Simple CLI interface to choose the seed for the RNG, the model to use (faces, celebs or cats) and the output directory.
- Random fallback seed if no seed is provided
- Customized output filename based on the seed and the model
- Added a Pip environment

## Get the trained models

To actually run the trained GAN, you need the pretrained models (duh). These are available in a Google Drive folder linked to from the [stylegan repo](https://github.com/NVlabs/stylegan). I had some issues downloading the files as Google kept saying that the quota was full, but I was able to just download the whole `networks` folder from Google Drive without tripping the quota limit. YMMV.

Once you have the data, find the files named something like "karras2019stylegan-\*\*\*-\*\*\*.pkl". These are the models, and you need to have them in the root of this project to run the GAN. You can also change the location where the script looks for these in the config.py file. I'm not sure how Nvidia would feel about me redistributing these trained models, but I'll err on the side of caution on this one and keep them gitignored.

Alternatively, if you have a bunch of professional Tesla GPU's sitting around, follow the instructions in the stylegan repo to train your own models. My GTX 1070 is good, but it ain't that good.

## Prerequisites

You need these to even attempt this:

- An nvidia graphics card with compute capability over 3.5 (I have a GTX 1070)
- Drivers for that
- Cuda and Cudnn (Cuda 10 should work fine)
- Python 3.7
- Pip and Pipenv

Install Nvidia stuff:

```bash
sudo pacman -S nvidia nvidia-utils cuda cudnn
```

Add Cuda to your path. Cuda installs in /opt/cuda on Arch. This is what I added in my zshrc:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/cuda/lib64:/opt/cuda/extras/CUPTI/lib64"
export PATH="$PATH:/opt/cuda/bin"
export CUDA_HOME=/opt/cuda
export NCCL_INSTALL_PATH="$CUDA_HOME/targets/x86_64-linux"
```

I had to mess around with Cuda a bit before it wanted to run. I'm not sure what I did but these are the basic requirements.

## Install

Install dependencies

```bash
pipenv install

```

## Run

To generate an image, run:

```bash
python simple.py
```

I have added a simple CLI that you can use to set a custom seed (it'll be random otherwise), the model you want to use (faces, celebs or cats) and the output directory:

```bash
python simple.py --model cats --output generated_images the_seed_you_want_to_use
```

If everything works correctly you'll have an imaginary cat in ./generated_images. You're welcome.