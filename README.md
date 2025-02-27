# llama_or_duck

## What does this project do?
This project is used to train a MobileNet V2 model (https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) that is used within a robot to play the iOS app Llama or Duck (https://apps.apple.com/us/app/llama-or-duck/id595311729).

https://github.com/user-attachments/assets/9f3e6fb7-fab3-4f9f-ab0b-08206348bf32

## Hardware used:
1. Raspberry Pi 4 Model B (4GB) with Raspberry Pi OS (64-bit) Debian Bookworm
2. Raspberry Pi Camera Modue V2
3. ULN2003 28BYJ-48 4-Phase Stepper Motor 5V Drive Board
4. Tenergy NiMH 6V 2000 mAh battery pack

![img1](https://github.com/user-attachments/assets/e9e6f467-2585-4cc2-81a0-2d7d7ca4b65c)
![img2](https://github.com/user-attachments/assets/d82b0a1c-95bc-41bb-891b-7e8e4284d410)

##  Images used for training:
Duck images
https://www.kaggle.com/datasets/alicenkbaytop/duck-images ,
https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

Llama images
https://www.kaggle.com/datasets/sid4sal/alpaca-dataset-small ,
https://www.kaggle.com/datasets/shivamaggarwal513/dlai-alpaca-dataset

## How can users get started with the project?
1. Clone this repository.
2. Change to llama_or_duck directory.
3. Create project venv with system packages (to use picamera2 and RPi libraries to control camera and motor) and activate.
     - python3 -m venv --system-site-packages venv
     - source venv/bin/activate
4. Install torch and torchvision with pip.
     - venv/bin/pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
5. Install compatible numpy version.
     - venv/bin/pip install numpy==1.26.4

To train:
1. Make a directory named "model_input_dir" with subdirectories named "llama" and "duck".
2. Put images into respective folders.
3. Change the TRAIN_DIR variable in the src/modeling/training/config.py file.
4. Run the src/modeling/training/entry.py file.

To play game:
1. Change to src directory.
2. Run main.py file.
