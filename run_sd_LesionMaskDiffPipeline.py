import argparse
import pprint
from typing import List
import torch
from PIL import Image
from LesionMaskDiffPipeline import LesionMaskDiffPipeline
from utils.lesion_mask_inference.ptp_utils import AttentionStore
from pathlib import Path
from typing import Dict, List
import os
from diffusers import UNet2DConditionModel

from utils.lesion_mask_inference.CrossAttnMap import AttentionStore as CustomAttentionStore
from utils.lesion_mask_inference.CustomAttnProcessor_for_inference import CustomDiffusionAttnProcessor

from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def load_model(config):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"

    unet = UNet2DConditionModel.from_pretrained(config.pretrained_unet_path, torch_dtype=torch.float32)

    controller = CustomAttentionStore(LOW_RESOURCE=False)
    controller.num_att_layers = 32

    # register ateention layers for recording, this is for prettrained unet
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


    stable = LesionMaskDiffPipeline.from_pretrained(stable_diffusion_version, unet=unet).to(device)

    stable.load_textual_inversion(config.pretrained_unet_path, weight_name="<lesion>.bin")
    stable.load_textual_inversion(config.pretrained_unet_path, weight_name="<skin>.bin")


    print(f"Loaded model: {stable_diffusion_version}")

    return stable, controller



def run_on_prompt(prompt: List[str],
                  model: LesionMaskDiffPipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  semantic_mask_cross: List[torch.Tensor],
                  semantic_mask_self: List[torch.Tensor],
                  prompt_output_path_image,
                  prompt_output_path_attention,
                  prompt_output_path_otsu,
                  config) -> Image.Image:

    for mask_id, mask in semantic_mask_cross.items():

        controller.cur_step = 0
        controller.cur_att_layer = 0
        controller.attention_store = controller.get_empty_store()

        self_mask =  semantic_mask_self[mask_id]

        outputs, target_token_attention_maps, transient_attention_map, transient_otsu_map = model(prompt=prompt,
                        attention_store=controller,
                        indices_to_alter=token_indices,
                        attention_res=config.attention_res,
                        self_attention_res=config.self_attention_res,
                        guidance_scale=config.guidance_scale,
                        generator=seed,
                        num_inference_steps=config.n_inference_steps,
                        max_iter_to_alter=config.max_iter_to_alter,
                        run_standard_sd=config.run_standard_sd,
                        scale_factor=config.scale_factor,
                        scale_range=config.scale_range,
                        sd_2_1=config.sd_2_1,
                        semantic_mask_cross=mask,
                        semantic_mask_self=self_mask,
                        config=config)
        image = outputs.images[0]


        mask_id = mask_id.split('.')[0]
        image.save(prompt_output_path_image / f'mask_{mask_id}.png')
        transient_attention_map[49].save(prompt_output_path_attention / f'mask_{mask_id}.png')
        transient_otsu_map[49].save(prompt_output_path_otsu / f'mask_{mask_id}.png')

def process_semantic_mask(mask_folder, size):
    image_transform = transforms.Compose([
        transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR), # the size of mask should be adjusted according to the res of attention map
        transforms.ToTensor()
    ])
    masks_path = Path(mask_folder)
    masks = {}
    for mask_path in masks_path.iterdir():
        mask_id  = mask_path.name.split('_')[1] # for ISIC dataset
        mask = Image.open(mask_path).convert('L')
        mask = image_transform(mask).to('cuda')
        masks[mask_id] = mask
    return masks



def main(config):
    stable, controller = load_model(config)
    token_indices = config.token_indices

    semantic_mask_cross = process_semantic_mask(config.semantic_mask_folder, config.attention_res)
    semantic_mask_self = process_semantic_mask(config.semantic_mask_folder, config.self_attention_res)

    for seed in config.seeds:
        print(f"Current seed is : {seed}")
        g = torch.Generator('cuda').manual_seed(seed)

        prompt_output_path = config.output_path / config.prompt[:100] / f'seed_{seed}'
        prompt_output_path.mkdir(exist_ok=True, parents=True)

        prompt_output_path_image = Path(prompt_output_path / 'image')
        prompt_output_path_image.mkdir(exist_ok=True, parents=True)

        prompt_output_path_attention = Path(prompt_output_path / 'attention')
        prompt_output_path_attention.mkdir(exist_ok=True, parents=True)

        prompt_output_path_otsu = Path(prompt_output_path / 'otsu')
        prompt_output_path_otsu.mkdir(exist_ok=True, parents=True)

        run_on_prompt(prompt=config.prompt,
                              model=stable,
                              controller=controller,
                              token_indices=token_indices,
                              seed=g,
                              semantic_mask_cross=semantic_mask_cross,
                              semantic_mask_self=semantic_mask_self,
                              prompt_output_path_image=prompt_output_path_image,
                              prompt_output_path_attention=prompt_output_path_attention,
                              prompt_output_path_otsu=prompt_output_path_otsu,
                              config=config)





def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SD BoxDiff for skin lesion generation with semantic masks")
    
    # Guiding text prompt
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="An image of <lesion> on <skin>",
        help="Guiding text prompt"
    )
    
    # Whether to use Stable Diffusion v2.1
    parser.add_argument(
        "--sd_2_1", 
        action="store_true", 
        default=True,
        help="Whether to use Stable Diffusion v2.1"
    )
    
    # Which token indices to alter with attend-and-excite
    parser.add_argument(
        "--token_indices", 
        type=int, 
        nargs="+", 
        default=[4],
        help="Which token indices to alter with attend-and-excite"
    )
    
    # Which random seeds to use when generating
    parser.add_argument(
        "--seeds", 
        type=int, 
        nargs="+", 
        default=[42],
        help="Which random seeds to use when generating"
    )
    
    # Path to save all outputs to
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./outputs_skin_lesion/",
        help="Path to save all outputs to"
    )
    
    # Number of denoising steps
    parser.add_argument(
        "--n_inference_steps", 
        type=int, 
        default=50,
        help="Number of denoising steps"
    )
    
    # Text guidance scale
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=7.5,
        help="Text guidance scale"
    )
    
    # Number of denoising steps to apply attend-and-excite
    parser.add_argument(
        "--max_iter_to_alter", 
        type=int, 
        default=None,
        help="Number of denoising steps to apply attend-and-excite (defaults to n_inference_steps * 0.5)"
    )
    
    # Resolution of UNet to compute attention maps over
    parser.add_argument(
        "--attention_res", 
        type=int, 
        default=16,
        help="Resolution of UNet to compute attention maps over"
    )
    
    # Resolution of UNet to compute self attention maps over
    parser.add_argument(
        "--self_attention_res", 
        type=int, 
        default=32,
        help="Resolution of UNet to compute self attention maps over"
    )
    
    # Whether to run standard SD or attend-and-excite
    parser.add_argument(
        "--run_standard_sd", 
        action="store_true", 
        default=False,
        help="Whether to run standard SD or attend-and-excite"
    )
    
    # Scale factor for updating the denoised latent z_t
    parser.add_argument(
        "--scale_factor", 
        type=int, 
        default=20,
        help="Scale factor for updating the denoised latent z_t"
    )
    
    # Start and end values used for scaling the scale factor
    parser.add_argument(
        "--scale_range", 
        type=float, 
        nargs=2, 
        default=[1.0, 0.5],
        help="Start and end values used for scaling the scale factor"
    )
    
    # Skin lesion model path
    parser.add_argument(
        "--pretrained_unet_path", 
        type=str, 
        default="/directory/to/pretrained/unet/model",
        help="Path to pretrained UNet model"
    )
    
    # Mask folder path
    parser.add_argument(
        "--semantic_mask_folder", 
        type=str, 
        default="/directory/to/semantic/mask/folder",
        help="Path to semantic mask folder"
    )
    
    args = parser.parse_args()
    
    # Convert output_path to Path object
    args.output_path = Path(args.output_path)
    
    # Set max_iter_to_alter if not provided
    if args.max_iter_to_alter is None:
        args.max_iter_to_alter = int(args.n_inference_steps * 0.5)
    
    # Convert scale_range to tuple
    args.scale_range = tuple(args.scale_range)
    
    return args


if __name__ == '__main__':
    config = parse_args()
    main(config)
