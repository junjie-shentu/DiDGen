
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
from torch.nn import functional as F
import cv2
from PIL import Image
from diffusers.utils import logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from utils.lesion_mask_inference.CrossAttnMap import AttentionStore, aggregate_current_attention, show_cross_attention_specific_token, view_images
from skimage import img_as_float
from skimage.restoration import denoise_bilateral
logger = logging.get_logger(__name__)

class LesionMaskDiffPipeline(StableDiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor"]

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds

    def _compute_max_attention_per_index(self,
                                         attention_maps: torch.Tensor,
                                         self_attention_maps: torch.Tensor,
                                         indices_to_alter: List[int],
                                         normalize_eot: bool = False,
                                         semantic_mask_cross: Union[torch.Tensor, None] = None,
                                         semantic_mask_self: Union[torch.Tensor, None] = None,
                                         config=None,
                                         ) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx] 
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        self_attention_map_size = self_attention_maps.shape[1] 

        self_attention_maps_for_text = self_attention_maps.reshape(self_attention_map_size**2, self_attention_map_size**2)#(32**2, 32**2)
        self_attention_maps_for_text = torch.matrix_power(self_attention_maps_for_text, 4)
        self_attention_maps_for_text = (self_attention_maps_for_text - self_attention_maps_for_text.min()) / (self_attention_maps_for_text.max() - self_attention_maps_for_text.min())

        self_attention_maps = self_attention_maps.reshape(self_attention_map_size, self_attention_map_size, self_attention_map_size, self_attention_map_size)

        if type(semantic_mask_cross) == torch.Tensor:
            semantic_mask_cross = [semantic_mask_cross]

        if type(semantic_mask_self) == torch.Tensor:
            semantic_mask_self = [semantic_mask_self]

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        assert len(semantic_mask_cross) == len(indices_to_alter), "Number of semantic masks should match the number of indices to alter"

        # Extract the maximum values
        max_indices_list_fg = []
        max_indices_list_bg = []
        dist_x = []
        dist_y = []

        loss_total = []

        cnt = 0
        for i, mask, mask_self in zip(indices_to_alter, semantic_mask_cross, semantic_mask_self):
            image = attention_for_text[:, :, i]

            # Binarize the mask
            mask_self = (mask_self > 0.5).float()

            loss_fg = 1 - torch.sum(image * mask) / torch.sum(mask)
            loss_bg = torch.sum(image * (1 - mask)) / torch.sum(1 - mask)

            loss_sar = []
            for i in range(mask_self.shape[1]):
                for j in range(mask_self.shape[2]):
                    if mask_self[:, i, j] > 0:
                        point_wise_self_attention_map = self_attention_maps[i, j, :, :]
                        point_wise_self_attention_map_background = point_wise_self_attention_map * (1 - mask_self)
                        loss_p = torch.sum(point_wise_self_attention_map_background)/torch.sum(1 - mask_self)
                        loss_sar.append(loss_p)

            loss_sar = torch.mean(torch.stack(loss_sar), dim=0)

            loss_total.append(loss_fg + loss_bg + loss_sar)


        return torch.mean(torch.stack(loss_total), dim=0), image, self_attention_maps_for_text




    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore,
                                                   indices_to_alter: List[int],
                                                   attention_res: int = 16,
                                                   self_attention_res: int = 32,
                                                   prompts: List[int] = None,
                                                   normalize_eot: bool = False,
                                                   semantic_mask_cross: Union[torch.Tensor, None] = None,
                                                   semantic_mask_self: Union[torch.Tensor, None] = None,
                                                   config=None,
                                                   ):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        

        attention_maps = aggregate_current_attention(
            prompts=prompts,
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0
        )

        self_attention_maps = aggregate_current_attention(
            prompts=prompts,
            attention_store=attention_store,
            res=self_attention_res,
            from_where=("up", "down", "mid"),
            is_cross=False,
            select=0
        )
        


        #max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._compute_max_attention_per_index(
        loss, cross_attention_map, self_attention_map = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            self_attention_maps=self_attention_maps,
            indices_to_alter=indices_to_alter,
            normalize_eot=normalize_eot,
            semantic_mask_cross=semantic_mask_cross,
            semantic_mask_self=semantic_mask_self,
            config=config,
        )
        return loss, cross_attention_map, self_attention_map #max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y

    @staticmethod
    def _compute_loss(max_attention_per_index_fg: List[torch.Tensor], max_attention_per_index_bg: List[torch.Tensor],
                      dist_x: List[torch.Tensor], dist_y: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses_fg = [max(0, 1. - curr_max) for curr_max in max_attention_per_index_fg]
        losses_bg = [max(0, curr_max) for curr_max in max_attention_per_index_bg]
        loss = sum(losses_fg) + sum(losses_bg) + sum(dist_x) + sum(dist_y)
        if return_losses:
            return max(losses_fg), losses_fg
        else:
            return max(losses_fg), loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           indices_to_alter: List[int],
                                           loss_fg: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           text_input,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           attention_res: int = 16,
                                           max_refinement_steps: int = 20,
                                           normalize_eot: bool = False,
                                           config=None,
                                           ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss_fg > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                normalize_eot=normalize_eot,
                config=config,
                )

            loss_fg, losses_fg = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y, return_losses=True)

            if loss_fg != 0:
                latents = self._update_latent(latents, loss_fg, step_size)

            with torch.no_grad():
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses_fg])
            except Exception as e:
                print(e)  # catch edge case :)

                low_token = np.argmax(losses_fg)

            if iteration >= max_refinement_steps:
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            normalize_eot=normalize_eot,
            config=config,
        )
        loss_fg, losses_fg = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y, return_losses=True)
        return loss_fg, latents, max_attention_per_index_fg

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 16,
            self_attention_res: int = 32,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.5),
            sd_2_1: bool = False,
            semantic_mask_cross: Union[torch.Tensor, None] = None,
            semantic_mask_self: Union[torch.Tensor, None] = None,
            config = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1


        # initialize the back-up attention store 
        # => because we use unet twice during a single denoising step: (1) predict 'noise_pred_text' to get 'region_loss' to update 'latents'; (2) predict 'noise_pred' to denoise and move to next step
        # so we set 'attention_store_backup'
        attention_store_backup = attention_store.get_empty_store()
        cur_step_backup = 0

        transient_attention_map = {}
        transient_otsu_map = {}

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                with torch.enable_grad():

                    latents = latents.clone().detach().requires_grad_(True)

                    # Forward pass of denoising with text conditioning
                    noise_pred_text = self.unet(latents, t,
                                                encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                    self.unet.zero_grad()

                    region_loss, cross_attention_map_cur_step, self_attention_map_cur_step = self._aggregate_and_get_max_attention_per_token(
                        attention_store=attention_store,
                        indices_to_alter=indices_to_alter,
                        attention_res=attention_res,
                        self_attention_res=self_attention_res,
                        prompts=text_inputs.input_ids,
                        normalize_eot=sd_2_1,
                        semantic_mask_cross=semantic_mask_cross,
                        semantic_mask_self=semantic_mask_self,
                        config=config,
                    )

                    if not run_standard_sd:

                        # Perform gradient update
                        if i < max_iter_to_alter:
                            if region_loss != 0:
                                latents = self._update_latent(latents=latents, loss=region_loss, #loss,
                                                              step_size=scale_factor * np.sqrt(scale_range[i]))



                if i in [49]:
                    cross_attention_map_cur_step = F.interpolate(cross_attention_map_cur_step.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2), (self_attention_res, self_attention_res), mode='bicubic', align_corners=False).permute(0, 2, 3, 1).squeeze(0).squeeze(-1)
                    self_cross_attention_map = (self_attention_map_cur_step @ cross_attention_map_cur_step.reshape(self_attention_res**2, 1)).reshape(self_attention_res, self_attention_res)
                    image_norm = (self_cross_attention_map - self_cross_attention_map.min()) / (self_cross_attention_map.max() - self_cross_attention_map.min())

                    image_norm = 255*image_norm
                    image_norm = image_norm.unsqueeze(-1).expand(*image_norm.shape, 3).cpu()
                    image_norm = image_norm.numpy().astype(np.uint8)
                    image_norm = np.array(Image.fromarray(image_norm).resize((512, 512)))
                    image_norm_pil_img = view_images(image_norm)

                    transient_attention_map[i] = image_norm_pil_img

                    # Apply Otsu's threshold
                    image_gray = cv2.cvtColor(image_norm, cv2.COLOR_RGB2GRAY)
                    _, binary_mask_otsu = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    binary_mask_otsu = np.expand_dims(binary_mask_otsu, axis=-1).repeat(3, axis=-1)
                    binary_mask_otsu = np.array(Image.fromarray(binary_mask_otsu).resize((512, 512)))
                    binary_mask_otsu_pil_img = view_images(binary_mask_otsu)
                    transient_otsu_map[i] = binary_mask_otsu_pil_img

                attention_store.reset()

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                attention_store.restore_attention_store(attention_store_backup, cur_step_backup)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                attention_store_backup, cur_step_backup = attention_store.back_up_attention_store()
                attention_store.reset()

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # get attention maps
        attention_store.restore_attention_store(attention_store_backup, cur_step_backup)
        target_token_attention_maps = show_cross_attention_specific_token(
            token_pos=indices_to_alter[0],
            prompts=text_inputs.input_ids,
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            select=0,
        )

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), target_token_attention_maps, transient_attention_map, transient_otsu_map
