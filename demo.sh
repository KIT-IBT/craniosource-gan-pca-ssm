#!/usr/bin/env bash

# PCA
python3 src/create_pca_images.py --number=10 --output_dir="./demo_out/pca" --input_dir=./dataset28/validation

# SSM
python3 src/create_ssm_images.py --number=10 --output_dir="./demo_out/ssm" --input_dir=./ssm

# GAN
python3 src/create_gan_images.py --number=10 --out_dir="./demo_out/gan" --generator_path=gan/generator.pt
