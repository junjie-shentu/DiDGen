"""
Inference script for generating skin lesion images with attention visualization.

This script generates images using a fine-tuned Stable Diffusion model and
visualizes cross-attention maps to understand model behavior.
"""

import argparse
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import yaml

import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
    UNet2DConditionModel,
)
from PIL import Image
from torchvision.transforms import ToPILImage

from utils.CrossAttnMap import AttentionStore, show_all_cross_attention_maps
from utils.CustomAttnProcessor import CustomDiffusionAttnProcessor

# Constants
DEFAULT_NUM_ATT_LAYERS = 16
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_ATTENTION_RESOLUTION = 16
DEFAULT_ATTENTION_LAYERS = ["down", "mid", "up"]




def register_unet(unet: UNet2DConditionModel) -> Tuple[UNet2DConditionModel, AttentionStore]:

    controller = AttentionStore(LOW_RESOURCE=False)
    controller.num_att_layers = DEFAULT_NUM_ATT_LAYERS

    custom_diffusion_attn_procs = {}

    for name, _ in unet.attn_processors.items():
        custom_diffusion_attn_procs[name] = CustomDiffusionAttnProcessor(
            train_k=False,
            train_v=False,
            train_q=False,
            train_out=False,
            hidden_size=None,
            cross_attention_dim=None,
            controller=controller,
            place_in_unet=name.split("_")[0],
        )

    unet.set_attn_processor(custom_diffusion_attn_procs)
    return unet, controller



def integrate_images_horizontally(img1: Image.Image, img2: Image.Image) -> Image.Image:

    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)

    # Step 4: Create a new image with the combined width and max height
    new_img = Image.new('RGB', (total_width, max_height))

    # Step 5: Paste img1 at the start
    new_img.paste(img1, (0, 0))

    # Step 6: Paste img2 next to img1
    new_img.paste(img2, (img1.width, 0))

    # Step 7: Save or display the new image
    return new_img




def load_pipeline(
    args: argparse.Namespace,
    weight_dtype: torch.dtype,
) -> Tuple[StableDiffusionPipeline, AttentionStore]:
    
    unet = UNet2DConditionModel.from_pretrained(args.ckpt_dir, torch_dtype=weight_dtype)
    unet, controller = register_unet(unet)

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    # Load textual inversion weights for <skin> and <lesion> tokens
    pipeline.load_textual_inversion(args.ckpt_dir, weight_name="<lesion>.bin")
    pipeline.load_textual_inversion(args.ckpt_dir, weight_name="<skin>.bin")



    # Configure scheduler and move to GPU
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=False)

    return pipeline, controller
        


def run_validation(
    args: argparse.Namespace,
    weight_dtype: torch.dtype,
    is_final_validation: bool = False,
) -> None:
    
    # Load pipeline
    pipeline, controller = load_pipeline(args, weight_dtype)
    
    device = torch.device("cuda")
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")
    
    for seed_idx, seed in enumerate(range(args.seed, args.seed + args.seed_range), 1):
        
        generator = torch.Generator(device=device).manual_seed(seed)

        for validation_prompt in args.validation_prompt:
            with inference_ctx:
                # Clean controller after each sampling step
                controller.cur_step = 0
                controller.cur_att_layer = 0
                controller.attention_store = controller.get_empty_store()

                # Generate image
                image = pipeline(
                    validation_prompt,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    num_images_per_prompt=args.num_validation_images
                ).images

                # Generate attention map
                attention_map = show_all_cross_attention_maps(
                    tokenizer=pipeline.tokenizer,
                    prompts=validation_prompt,
                    attention_store=controller,
                    res=args.attention_resolution,
                    from_where=args.attention_layers,
                    num_prompts=1
                )[0]

            # Create filename and save combined image
            caption = f"{validation_prompt}_{seed_idx}.png"
            grid = integrate_images_horizontally(image[0], attention_map)
            
            output_path = os.path.join(args.image_save_path, caption)
            grid.save(output_path)




def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate skin lesion images with attention visualization"
    )
    
    # Model configuration
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path to pretrained model or model identifier"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to the output directory containing model checkpoints"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model variant"
    )
    
    # Generation parameters
    parser.add_argument(
        "--validation_prompt",
        type=str,
        nargs="+",
        default=["An image of <lesion> on <skin>"],
        help="Validation prompts for image generation"
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation"
    )
    parser.add_argument(
        "--seed_range",
        type=int,
        default=20,
        help="Number of different seeds to use"
    )
    
    # Attention visualization parameters
    parser.add_argument(
        "--attention_resolution",
        type=int,
        default=DEFAULT_ATTENTION_RESOLUTION,
        help="Resolution for attention maps"
    )
    parser.add_argument(
        "--attention_layers",
        type=str,
        nargs="+",
        default=DEFAULT_ATTENTION_LAYERS,
        help="UNet layers to visualize attention from"
    )
    
    # Output configuration
    parser.add_argument(
        "--image_save_path",
        type=str,
        required=True,
        help="Path to save generated images"
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights"
    )
    parser.add_argument(
        "--is_final_validation",
        action="store_true",
        help="Run final validation (disables autocast)"
    )

    
    args = parser.parse_args()

    
    return args


def main() -> None:
    """Main function."""
    args = parse_args()
    
    # Convert weight dtype
    weight_dtype = getattr(torch, args.weight_dtype)
    
    # Create output directory
    os.makedirs(args.image_save_path, exist_ok=True)
    
    # Run validation
    run_validation(
        args,
        weight_dtype,
        is_final_validation=args.is_final_validation
    )


if __name__ == "__main__":
    main()