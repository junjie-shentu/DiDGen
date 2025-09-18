import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from torchvision.transforms import ToPILImage
import json
import itertools

from utils.CustomAttnProcessor import CustomDiffusionAttnProcessor
from utils.CustomModelLoader2 import CustomModelLoader
from utils.CrossAttnMap import AttentionStore, aggregate_current_attention
from diffusers.loaders import AttnProcsLayers

import safetensors
import pandas as pd


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.2")

logger = get_logger(__name__)

def save_new_embed(text_encoder, modifier_token_id, accelerator, args, output_dir, safe_serialization=True):
    """Saves the new token embeddings from the text encoder."""
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
    for x, y in zip(modifier_token_id, args.modifier_token):
        learned_embeds_dict = {}
        learned_embeds_dict[y] = learned_embeds[x]
        filename = f"{output_dir}/{y}.bin"

        if safe_serialization:
            safetensors.torch.save_file(learned_embeds_dict, filename, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict, filename)


def find_the_identify_token_index(tokens, modifier_token_id):
    """
    The structure of the prompt is like: A photo of ... and ...
    So we need to find the index of the first "and" token, which is the token before the second object.

    Update: from this version we start using subset sampling, so we need to check if and_index is empty.
    """
    target_indices = [token_id for token_id, token in enumerate(tokens) if token in modifier_token_id]

    return target_indices


def show_attention_map_during_training(cross_attention_map, obj, global_step):
    split_tensors = torch.split(cross_attention_map, 1, dim=0)

    # create a list to store the PIL images
    images = []

    # loop over the split tensors and show the PIL image of each element
    for i, tensor in enumerate(split_tensors):
        # convert the tensor to a PIL image
        image = Image.fromarray(tensor.squeeze().mul(255).clamp(0, 255).byte().cpu().numpy())

        # append the image to the list
        images.append(image)

    # combine the images into a 1 row 4 column image
    combined_image = Image.new(mode="RGB", size=(cross_attention_map.shape[1] * len(split_tensors), cross_attention_map.shape[2]))
    for i, image in enumerate(images):
        combined_image.paste(image, (i * cross_attention_map.shape[1], 0))

    # log the combined image to wandb
    #wandb.log({f"cross_attention_maps_obj_{obj}": wandb.Image(combined_image)})
    wandb.log({f"{obj}": wandb.Image(combined_image)}, step=global_step)

def freeze_params(params):
    for param in params:
        param.requires_grad = False


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid



def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="skin_lesion_generation",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default="pretrain_unet_with_attention.json",
        help=(
            "The json file that specify data source."
        ),
    )
    parser.add_argument("--with_prior_preservation", action="store_true")
    parser.add_argument("--train_k", action="store_true")
    parser.add_argument("--train_v", action="store_true")
    parser.add_argument("--train_q", action="store_true")
    parser.add_argument("--train_out", action="store_true")
    parser.add_argument("--num_class_images", type=int, default=1000)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--modifier_token", type=str, default="<lesion>+<skin>")
    parser.add_argument("--initializer_token", type=str, default="lesion+skin")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


    

class Skindataset(torch.utils.data.Dataset):
    def __init__(self, concept_list, tokenizer, size=512, with_prior_preservation=True, num_class_images=200):

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation

        for concept in concept_list:
            with open(concept['instance_prompt_json'], "r") as f:
                self.instance_prompts = json.load(f)

            instance = [(x, self.instance_prompts[x.name]["content"], os.path.join(concept["segmask_dir"], (x.stem + "_segmentation.png"))) for x in Path(concept["instance_data_dir"]).iterdir() if x.suffix in ['.jpg', '.png']]
            self.instance_images_path.extend(instance)

            if self.with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [image_path.name.split("-")[0] for image_path in class_images_path]
                    #class_images_path = [(x, concept["class_prompt"], os.path.join(concept["class_segmask_dir"], (x.stem + "_segmentation.png"))) for x in Path(concept["class_data_dir"]).iterdir() if x.suffix in ['.jpg', '.png']]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()
            
                class_img_path = list(zip(class_images_path, class_prompt))
                self.class_images_path.extend(class_img_path[:num_class_images])
                #self.class_images_path.extend(class_images_path[:num_class_images])

        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)


        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]
        )

        self.tokenizer = tokenizer


    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        example = {}
        instance_image, instance_prompt, mask = self.instance_images_path[idx % self.num_instance_images]
        instance_image = Image.open(instance_image).convert("RGB")
        mask = Image.open(mask).convert("L")

        example["pixel_values"] = self.image_transforms(instance_image)
        example["conditioning_pixel_values"] = self.conditioning_image_transforms(mask)

        instance_prompt = instance_prompt.replace("lesion", "<lesion>").replace("skin", "<skin>")

        num_determiner_1 = random.random()
        if num_determiner_1 < 0.2:
            instance_prompt =  "An image of <lesion> on <skin>."
        else:
            instance_prompt =  "An image of <lesion> on <skin>." + " " + instance_prompt
        example["input_ids"] = self.tokenizer(instance_prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

        example["attention_mask"] = (example["conditioning_pixel_values"], torch.ones_like(example["conditioning_pixel_values"]) - example["conditioning_pixel_values"])

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[idx % self.num_class_images]
            class_image = Image.open(class_image).convert("RGB")

            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(class_prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

        return example



def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["input_ids"] for example in examples]
    pixel_values = [example["pixel_values"] for example in examples]
    attention_mask = [example["attention_mask"] for example in examples]

    

    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    input_ids = torch.cat(input_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    attention_mask = torch.stack(attention_mask)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }



def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "segmask_dir":args.segmask_dir,
                "class_data_dir": args.class_data_dir,
                "class_segmask_dir": args.class_segmask_dir,
                "learning_target":args.learning_target
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    if args.with_prior_preservation and args.concepts_list[0]["class_data_dir"] is None:
        raise ValueError("You need to provide a `class_data_dir` to use `--with_prior_preservation`")


    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )



    
    ########################################################
    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    initializer_token_id = []
    if args.modifier_token is not None:
        args.modifier_token = args.modifier_token.split("+")
        args.initializer_token = args.initializer_token.split("+")
        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(
            args.modifier_token, args.initializer_token[: len(args.modifier_token)]
        ):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
            print(token_ids)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) == 1: 
                token_ids = token_ids
            elif len(token_ids) > 1:
                token_ids = token_ids[:1]

            initializer_token_id.append(token_ids)
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for x, y in zip(modifier_token_id, initializer_token_id):
            if len(y) == 1:
                token_embeds[x] = token_embeds[y[0]]
            else:
                token_embeds[x] = torch.mean(token_embeds[y], dim=0)

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
            text_encoder.get_input_embeddings().parameters() # also freeze the newly added token embeddings, since we don't want to update the text embedding of modifier token in this version
        )
        freeze_params(params_to_freeze)
    ########################################################



    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    sub_dir = "unet" if isinstance(model, type(unwrap_model(unet))) else "text_encoder"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()


        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                if isinstance(model, type(unwrap_model(text_encoder))):
                    # load transformers style into model
                    load_model = text_encoder_cls.from_pretrained(input_dir, subfolder="text_encoder")
                    model.config = load_model.config
                else:
                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(True)
    if args.modifier_token is None:
        text_encoder.requires_grad_(False)


    train_k = args.train_k
    train_v = args.train_v
    train_q = args.train_q
    train_out = args.train_out
    custom_diffusion_attn_procs = {}

    controller = AttentionStore(LOW_RESOURCE=False)
    controller.num_att_layers = 32  

    st = unet.state_dict()
    num_cross_attn_layers = 0
    for name, _ in unet.attn_processors.items():
        cross_attention_dim = None
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        layer_name = name.split(".processor")[0]
        

        weights = {}
        if train_k:
            weights["to_k_custom_diffusion.weight"] = st[layer_name + ".to_k.weight"]
        if train_v:
            weights["to_v_custom_diffusion.weight"] = st[layer_name + ".to_v.weight"]
        if train_q:
            weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
        if train_out:
            weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
            weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]

        if cross_attention_dim is not None:
            custom_diffusion_attn_procs[name] = CustomDiffusionAttnProcessor(
                train_k=train_k,
                train_v=train_v,
                train_q=train_q,
                train_out=train_out,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                controller=controller,
                place_in_unet=name.split("_")[0],
            ).to(unet.device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)

            num_cross_attn_layers += 1
        else:
            custom_diffusion_attn_procs[name] = CustomDiffusionAttnProcessor(
                train_k=False,
                train_v=False,
                train_q=False,
                train_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                controller=controller,
                place_in_unet=name.split("_")[0],
            )


    del st
    unet.set_attn_processor(custom_diffusion_attn_procs)
    custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)

    accelerator.register_for_checkpointing(custom_diffusion_layers)

    #############################################################################
    #register CustomModelLoader as model saving hook
    loader = CustomModelLoader(unet=unet)
    #############################################################################


    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = unet.parameters()

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    train_dataset = Skindataset(args.concepts_list, tokenizer, args.resolution, args.with_prior_preservation, args.num_class_images)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )


    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )


    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    from_where = ["up","down", "mid"]
    cross_res = 16

    

    def get_attention_maps(prompts, bsz, GT_attention_maps):#calculate_cross_attn_loss, mask_token, GT_attention_maps):

        attention_loss = 0.0
        attn_losses = []


        if args.with_prior_preservation:
            num_samples_to_extract_attention = bsz  // 2
        else:
            num_samples_to_extract_attention = bsz

        for i in range(num_samples_to_extract_attention):
            cross_attention_map = aggregate_current_attention(prompts=prompts,
                                                            attention_store=controller, 
                                                            res=cross_res, 
                                                            from_where=from_where,
                                                            is_cross=True,
                                                            select=i)

            GT_attention_maps_i = GT_attention_maps[i]

            target_indices = find_the_identify_token_index(prompts[i], modifier_token_id)[:2]


            for target_index in target_indices:
                cross_image = cross_attention_map[:, :, target_index]

                #normalization
                # create a new tensor to hold the result of the computation instead of directly modifying the original tensor to avoid in-place operation error
                cross_image_normalized = torch.zeros_like(cross_image)
                cross_image_normalized = (cross_image - torch.min(cross_image)) / (torch.max(cross_image) - torch.min(cross_image))

                # Interpolate GT_attention_maps_i to match the size of cross_image (16x16)
                GT_map = F.interpolate(GT_attention_maps_i[target_index].unsqueeze(0).unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
                loss = torch.nn.functional.mse_loss(cross_image_normalized, GT_map)
                attn_losses.append(loss)

            attention_loss = torch.mean(torch.stack(attn_losses), dim=0)

        return attention_loss
    




    # if we don't want to save the ckpt during training, we set the checkpointing steps to the max_train_steps
    if args.checkpointing_steps > args.max_train_steps:
        args.checkpointing_steps = args.max_train_steps

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                attention_mask = batch["attention_mask"].to(dtype=weight_dtype)


            
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Calculate the loss
                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)


                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                attention_loss = get_attention_maps(prompts = batch["input_ids"],
                                                    bsz = bsz,
                                                    GT_attention_maps = attention_mask)

                wandb.log({"denoise_loss": loss}, step=global_step)
                if args.with_prior_preservation:
                    wandb.log({"prior_loss": prior_loss}, step=global_step)
                wandb.log({"attention_loss": attention_loss}, step=global_step)

                #clean controller after each sampling or training step
                controller.cur_step = 0
                controller.cur_att_layer = 0
                controller.attention_store = controller.get_empty_store()


                total_loss = loss + 0.1 * attention_loss
                wandb.log({"total_loss": total_loss}, step=global_step)

                accelerator.backward(total_loss)


                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                ################################################



                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if global_step >= args.max_train_steps:
                            accelerator.save_state(save_path)
                        else:
                            unet.save_pretrained(save_path)
                        logger.info(f"Saved state to {save_path}")

                        #save new embedding
                        save_new_embed(
                            text_encoder,
                            modifier_token_id,
                            accelerator,
                            args,
                            save_path,
                            safe_serialization=False,
                        )



            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break



    accelerator.end_training()


if __name__ == "__main__":

    args = parse_args()
    main(args)