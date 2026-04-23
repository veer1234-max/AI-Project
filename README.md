Local Stable Diffusion Studio

Live Link - https://huggingface.co/spaces/Veerman1234/AiModel

A full-featured Stable Diffusion web interface built with Python, Gradio, Diffusers, and PyTorch. This project provides a clean browser-based workspace for loading models, generating images from text prompts, transforming existing images, inpainting, outpainting, upscaling, prompt interrogation, and model merging. It is designed for users who want a practical, all-in-one interface for experimenting with Stable Diffusion models without building a UI from scratch.

Features

Model Manager Load models from a Hugging Face repo id Load local checkpoints or model folders Refresh detected local models Unload models from memory Text to Image Generate images from prompts Control resolution, steps, CFG scale, image count, and seed Image to Image Upload an image and transform it with a text prompt Adjust denoising strength Inpainting Mask part of an image and regenerate selected areas Outpainting Expand the canvas and generate new surrounding content Extras Upscaling support Optional face restoration when supported tools are installed CLIP Interrogator / Captioning Inspect an image and generate a descriptive prompt Model Merge Merge compatible .safetensors checkpoints with weighted blending

Tech Stack

Python Gradio Diffusers PyTorch Transformers NumPy Pillow Hugging Face Hub

First-Time User Guide

When you open the project for the first time, follow these steps:

Step 1: Open the Model Manager tab

This is where you load a Stable Diffusion model before generating anything.

Step 2: Enter a model source

In the Model source field, enter a Hugging Face model id. ex- runwayml/stable-diffusion-v1-5

Step 3: Click Load model

Wait until the model finishes loading. The first load may take some time depending on hardware and internet speed.

Step 4: Go to Text to Image

After the model is loaded, switch to the Text to Image tab.

Step 5: Enter your prompt

Example:A cinematic portrait of a fox astronaut, ultra realistic, sharp focus, soft lighting, detailed fur, space suit

Step 6: Optional negative prompt blurry, low quality, deformed, bad anatomy, distorted face, extra limbs

Step 7: Choose generation settings

Recommended starting settings:

Width: 512 Height: 512 Steps: 15 CFG scale: 7.5 Images: 1 Seed: -1

These settings are a good balance between speed and quality, especially on CPU.

Step 8: Click Generate

Wait for the image to appear in the output gallery.

Step 9: Try other tabs

Once basic generation works, you can explore:

Image to Image for stylising uploaded images Inpainting for editing specific areas Outpainting for extending images Extras for upscaling CLIP Interrogator for prompt recovery Model Merge for combining checkpoints

Recommended Settings for New Users Fast setup

Best for slower machines or CPU environments:

512 × 512 10 to 15 steps CFG 6 to 7.5 1 image Better quality

If you have stronger hardware:

768 × 768 or higher 20 to 30 steps CFG 7 to 8 1 image Notes on Performance

Performance depends heavily on hardware.

On CPU Model loading can take time Image generation can be slow 512 × 512 is strongly recommended On GPU Much faster loading Higher resolutions are practical Better overall experience

If running on Hugging Face Spaces or another hosted environment, GPU hardware will significantly improve speed.
