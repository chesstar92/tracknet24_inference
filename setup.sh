#!/bin/bash

# init submodules (WASB, fast-volleyball-tracking-inference)
git submodule update --init --remote

# download pre-trained weights for TrackNetV2 volleyball
mkdir -p WASB-SBDT/pretrained_weights
wget https://drive.google.com/uc?id=103jOdYp4k20avid4uyB9USCuwiphI4Kz -O WASB-SBDT/pretrained_weights/tracknetv2_volleyball_best.pth.tar

# resolve python dependencies
uv sync
