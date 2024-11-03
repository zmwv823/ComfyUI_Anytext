import folder_paths
from typing import Optional, List, Union, Dict, Tuple, Callable, Any
import torch
import os
from transformers import T5EncoderModel, T5Tokenizer
import torch.nn.functional as F
from comfy.utils import ProgressBar
from comfy.model_management import text_encoder_offload_device
from .......UL_common.common import clean_up

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import (
    StableDiffusionXLInpaintPipeline,
    AutoencoderKL,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    UNet2DConditionModel,
    KarrasDiffusionSchedulers,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    VaeImageProcessor,
    # is_invisible_watermark_available,
    StableDiffusionXLLoraLoaderMixin,
    PipelineImageInput,
    adjust_lora_scale_text_encoder,
    scale_lora_layers,
    unscale_lora_layers,
    USE_PEFT_BACKEND,
    StableDiffusionXLPipelineOutput,
    ImageProjection,
    logging,
    rescale_noise_cfg,
    retrieve_timesteps,
    deprecate,
)
import numpy as np
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

class StableDiffusionGlyphXLPipeline_Inpaint(StableDiffusionXLInpaintPipeline):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->byt5_text_encoder->image_encoder->unet->byt5_mapper->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "byt5_tokenizer",
        "text_encoder",
        "text_encoder_2",
        "byt5_text_encoder",
        "byt5_mapper",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        byt5_text_encoder: T5EncoderModel,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        byt5_tokenizer: T5Tokenizer,
        byt5_mapper,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        byt5_max_length: int = 512,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        requires_aesthetics_score: bool = False,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.negative_pooled_prompt_embeds = None
        self.byt5_prompt_embeds = None
        self.negative_byt5_prompt_embeds = None
        
        super(StableDiffusionXLInpaintPipeline, self).__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            byt5_text_encoder=byt5_text_encoder,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            byt5_tokenizer=byt5_tokenizer,
            byt5_mapper=byt5_mapper,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)
        self.register_to_config(byt5_max_length=byt5_max_length)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.byt5_max_length = byt5_max_length

        self.default_sample_size = self.unet.config.sample_size

        # add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        self.watermark = add_watermarker
        # if add_watermarker:
        #     self.watermark = StableDiffusionXLWatermarker()
        # else:
        #     self.watermark = None

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )

            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd derivative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1

            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            timesteps = timesteps[-num_inference_steps:]
            return timesteps, num_inference_steps

        return timesteps, num_inference_steps - t_start

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        if masked_image is not None and masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = None

        if masked_image is not None:
            if masked_image_latents is None:
                masked_image = masked_image.to(device=device, dtype=dtype)
                masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            masked_image_latents = (
                torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
            )

            # aligning device to prevent device errors when concating it with the latent model input
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        return mask, masked_image_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None, image_encoder_type='Vit-H', dtype=torch.float16):
        if 'Vit' in image_encoder_type:
            from .......UL_common.singlefile_config_paths import CLIP_ViT_H_14_laion2B_s32B_b79K_config_path, CLIP_ViT_bigG_14_laion2B_39B_b160k_config_path
            from transformers import CLIPImageProcessor
            from transformers.configuration_utils import PretrainedConfig
            clip_Vit_image_processor = CLIPImageProcessor()
            from safetensors.torch import load_file as safetensor_loadfile
            if 'Vit-H' in image_encoder_type:
                state_dict = safetensor_loadfile(os.path.join(folder_paths.models_dir,'clip_vision','CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors'))
                Vit_clip_vision_config = PretrainedConfig.from_json_file(CLIP_ViT_H_14_laion2B_s32B_b79K_config_path)
            else:
                state_dict = safetensor_loadfile(os.path.join(folder_paths.models_dir,'clip_vision','CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors'))
                Vit_clip_vision_config = PretrainedConfig.from_json_file(CLIP_ViT_bigG_14_laion2B_39B_b160k_config_path)
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path=None, config=Vit_clip_vision_config, state_dict=state_dict).to(device, dtype)
            del state_dict
        else:
            from transformers import AutoImageProcessor, AutoModel
            dinov2_model_path = os.path.join(folder_paths.models_dir, 'clip_vision', 'models--facebook--dinov2-large')
            dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_model_path)
            image_encoder = AutoModel.from_pretrained(dinov2_model_path).to(device, dtype)
        
        if not isinstance(image, torch.Tensor):
            if 'Vit' in image_encoder_type:
                image = clip_Vit_image_processor(image, return_tensors="pt").pixel_values
            else:
                dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device, dtype)
                image = dinov2_inputs.pixel_values
                uncond_image = torch.zeros_like(image).to(device, dtype)

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            print('\033[93m', 'Output ipadapter hidden_states.', '\033[0m')
            if 'Vit' in image_encoder_type:
                image_enc_hidden_states = image_encoder(image, output_hidden_states=True).hidden_states[-2]
            else:
                image_enc_hidden_states = image_encoder(**dinov2_inputs).last_hidden_state
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            if 'Vit' in image_encoder_type:
                uncond_image_enc_hidden_states = image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[-2]
            else:
                uncond_image_enc_hidden_states = image_encoder(uncond_image).last_hidden_state
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            print('\033[93m', f'Image encode by {image_encoder_type} completed.', '\033[0m')
            del image_encoder
            clean_up()
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            print('\033[93m', 'Output image_embeds.', '\033[0m')
            image_embeds = image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)
            print('\033[93m', 'Image encode completed.', '\033[0m')

            del image_encoder
            clean_up()
            return image_embeds, uncond_image_embeds

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        text_prompt = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        text_attn_mask: Optional[torch.LongTensor] = None,
        byt5_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale
            print('\033[93m', f'Apply lora scale: {lora_scale}.', '\033[0m')

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            assert len(prompt) == 1
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            
            text_prompt = [text_prompt] if isinstance(text_prompt, str) else text_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            text_input_id_batchs = []
            for prompt, tokenizer in zip(prompts, tokenizers):
                pad_token = tokenizer.pad_token_id
                total_tokens = tokenizer(prompt, truncation=False)['input_ids'][0]
                bos = total_tokens[0]
                eos = total_tokens[-1]
                total_tokens = total_tokens[1:-1]
                new_total_tokens = []
                empty_flag = True
                while len(total_tokens) >= 75:
                    head_75_tokens = [total_tokens.pop(0) for _ in range(75)]
                    temp_77_token_ids = [bos] + head_75_tokens + [eos]
                    new_total_tokens.append(temp_77_token_ids)
                    empty_flag = False
                if len(total_tokens) > 0 or empty_flag:
                    padding_len = 75 - len(total_tokens)
                    temp_77_token_ids = [bos] + total_tokens + [eos] + [pad_token] * padding_len
                    new_total_tokens.append(temp_77_token_ids)
                # 1,segment_len, 77
                new_total_tokens = torch.tensor(new_total_tokens, dtype=torch.long).unsqueeze(0)
                text_input_id_batchs.append(new_total_tokens)
            if text_input_id_batchs[0].shape[1] > text_input_id_batchs[1].shape[1]:
                tokenizer = tokenizers[1]
                pad_token = tokenizer.pad_token_id
                bos = tokenizer.bos_token_id
                eos = tokenizer.eos_token_id
                padding_len = text_input_id_batchs[0].shape[1] - text_input_id_batchs[1].shape[1]
                # padding_len, 77
                padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
                # 1, padding_len, 77
                padding_part = padding_part.unsqueeze(0)
                text_input_id_batchs[1] = torch.cat((text_input_id_batchs[1],padding_part), dim=1)
            elif text_input_id_batchs[0].shape[1] < text_input_id_batchs[1].shape[1]:
                tokenizer = tokenizers[0]
                pad_token = tokenizer.pad_token_id
                bos = tokenizer.bos_token_id
                eos = tokenizer.eos_token_id
                padding_len = text_input_id_batchs[1].shape[1] - text_input_id_batchs[0].shape[1]
                # padding_len, 77
                padding_part = torch.tensor([[bos] + [eos] + [pad_token] * 75 for _ in range(padding_len)])
                # 1, padding_len, 77
                padding_part = padding_part.unsqueeze(0)
                text_input_id_batchs[0] = torch.cat((text_input_id_batchs[0],padding_part), dim=1)
            
            embeddings = []
            for segment_idx in range(text_input_id_batchs[0].shape[1]):
                prompt_embeds_list = []
                for i, text_encoder in enumerate(text_encoders):
                    # 1, segment_len, sequence_len
                    text_input_ids = text_input_id_batchs[i].to(text_encoder.device)
                    # 1, sequence_len, dim
                    prompt_embeds = text_encoder(
                        text_input_ids[:, segment_idx],
                        output_hidden_states=True,
                    )

                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    temp_pooled_prompt_embeds = prompt_embeds[0]
                    if clip_skip is None:
                        prompt_embeds = prompt_embeds.hidden_states[-2]
                    else:
                        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
                    bs_embed, seq_len, _ = prompt_embeds.shape
                    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                    prompt_embeds_list.append(prompt_embeds)
                # b, sequence_len, dim
                prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
                embeddings.append(prompt_embeds)
                if segment_idx == 0:
                    # use the first segment's pooled prompt embeddings as 
                    # the pooled prompt embeddings
                    # b, dim->b, dim
                    pooled_prompt_embeds = temp_pooled_prompt_embeds.view(bs_embed, -1)
            # b, segment_len * sequence_len, dim
            prompt_embeds = torch.cat(embeddings, dim=1)
            
            if byt5_prompt_embeds is None:
                byt5_text_inputs = self.byt5_tokenizer(
                    text_prompt,
                    padding="max_length",
                    max_length=self.byt5_max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                byt5_text_input_ids = byt5_text_inputs.input_ids
                byt5_attention_mask = byt5_text_inputs.attention_mask.to(self.byt5_text_encoder.device) if text_attn_mask is None else text_attn_mask.to(self.byt5_text_encoder.device, dtype=byt5_text_inputs.attention_mask.dtype)
                # with torch.cuda.amp.autocast(enabled=False):
                autocast_device_type = device.type
                with torch.amp.autocast(autocast_device_type, enabled=False):
                    byt5_prompt_embeds = self.byt5_text_encoder(
                        byt5_text_input_ids.to(self.byt5_text_encoder.device),
                        attention_mask=byt5_attention_mask.float(),
                    )
                    byt5_prompt_embeds = byt5_prompt_embeds[0]
                    byt5_prompt_embeds = self.byt5_mapper(byt5_prompt_embeds, byt5_attention_mask)

        # get unconditional embeddings for classifier free guidance
        # zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        zero_out_negative_prompt = negative_prompt is None or self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_byt5_prompt_embeds = torch.zeros_like(byt5_prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            raise NotImplementedError

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            byt5_seq_len = negative_byt5_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)
            negative_byt5_prompt_embeds = negative_byt5_prompt_embeds.to(dtype=self.byt5_text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            negative_byt5_prompt_embeds = negative_byt5_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_byt5_prompt_embeds = negative_byt5_prompt_embeds.view(batch_size * num_images_per_prompt, byt5_seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return (
            prompt_embeds, 
            negative_prompt_embeds, 
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            byt5_prompt_embeds, 
            negative_byt5_prompt_embeds,
        )

    @torch.no_grad()
    def __call__(
        self,
        device,
        debug,
        loaded_lora_scale,
        loaded_prompt_and_bboxes,
        loaded_embeds,
        loaded_attn_masks_dicts,
        loaded_ip_adapter_img,
        loaded_image_embeds,
        loaded_clip_vision,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.Tensor = None,
        text_prompt = None,
        texts = None,
        bboxes = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 0.9999,
        do_classifier_free_guidance: bool = True,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        text_attn_mask: torch.LongTensor = None,
        byt5_prompt_embeds: Optional[torch.FloatTensor] = None,
        image_encoder_type: str = None,
        **kwargs,
    ):
        dtype = self.unet.dtype
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            mask_image,
            width,
            height,
            strength,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
            padding_mask_crop,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._denoising_start = denoising_start
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        
        if debug:
            print('\033[93m', '-----Debug info in pipe-----', '\033[0m')
            print('\033[93m', f'prompt: {prompt}', '\033[0m')
            print('\033[93m', f'prompt_2: {prompt_2}', '\033[0m')
            print('\033[93m', f'text_prompt: {text_prompt}', '\033[0m')
            print('\033[93m', f'texts: {texts}', '\033[0m')
            print('\033[93m', f'do_classifier_free_guidance: {do_classifier_free_guidance}', '\033[0m')
            print('\033[93m', f'negative_prompt: {negative_prompt}', '\033[0m')
            print('\033[93m', f'negative_prompt_2: {negative_prompt_2}', '\033[0m')
            print('\033[93m', f'lora_scale: {lora_scale}', '\033[0m')
            print('\033[93m', f'self.clip_skip: {self.clip_skip}', '\033[0m')

        if loaded_prompt_and_bboxes != [prompt + text_prompt, bboxes] or loaded_lora_scale != lora_scale:
            loaded_lora_scale = lora_scale
            if debug:
                print('\033[93m', 'Start encode prompt.', '\033[0m')
            if 'cpu' in self.text_encoder.device.type:
                self.text_encoder.to(device, dtype)
                self.text_encoder_2.to(device, dtype)
                self.byt5_text_encoder.to(device)
                self.byt5_mapper.to(device)
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                byt5_prompt_embeds,
                negative_byt5_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                text_prompt=text_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                text_attn_mask=text_attn_mask,
                byt5_prompt_embeds=byt5_prompt_embeds,
            )
            self.prompt_embeds = prompt_embeds.to(device, dtype)
            self.negative_prompt_embeds = negative_prompt_embeds.to(device, dtype)
            self.pooled_prompt_embeds = pooled_prompt_embeds.to(device, dtype)
            self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device, dtype)
            self.byt5_prompt_embeds = byt5_prompt_embeds.to(device)
            self.negative_byt5_prompt_embeds = negative_byt5_prompt_embeds.to(device)
            loaded_embeds = [self.prompt_embeds, self.negative_prompt_embeds, self.pooled_prompt_embeds, self.negative_pooled_prompt_embeds, self.byt5_prompt_embeds, self.negative_byt5_prompt_embeds]
            if debug:
                print('\033[93m', 'Encode prompt completed.', '\033[0m')
        
            self.text_encoder.to(text_encoder_offload_device())
            self.text_encoder_2.to(text_encoder_offload_device())
            self.byt5_text_encoder.to(text_encoder_offload_device())
            self.byt5_mapper.to(text_encoder_offload_device())
        else:
            self.prompt_embeds, self.negative_prompt_embeds, self.pooled_prompt_embeds, self.negative_pooled_prompt_embeds, self.byt5_prompt_embeds, self.negative_byt5_prompt_embeds = loaded_embeds[0], loaded_embeds[1], loaded_embeds[2], loaded_embeds[3], loaded_embeds[4], loaded_embeds[5]

        # 4. Prepare timesteps
        # def denoising_value_valid(dnv):
        #     return isinstance(dnv, float) and 0 < dnv < 1
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        # timesteps, num_inference_steps = self.get_timesteps(
        #     num_inference_steps,
        #     strength,
        #     device,
        #     denoising_start=self.denoising_start if denoising_value_valid(self.denoising_start) else None,
        # )
        
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        init_image = self.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        init_image = init_image.to(dtype=torch.float32)

        mask = self.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )

        if masked_image_latents is not None:
            masked_image = masked_image_latents
        elif init_image.shape[1] == 4:
            # if images are in latent space, we can't mask it
            masked_image = None
        else:
            masked_image = init_image * (mask < 0.5)

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4
        
        if 'cpu' in self.vae.device.type:
            self.vae.to(device)

        add_noise = True if self.denoising_start is None else False
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            add_noise=add_noise,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents
        else:
            latents, noise = latents

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            self.prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )
        
        self.vae.to('cpu')

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        # 7. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size
            
        add_text_embeds = self.pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(self.pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=self.prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                target_size,
                aesthetic_score,
                negative_aesthetic_score,
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=self.prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            self.prompt_embeds = torch.cat([self.negative_prompt_embeds, self.prompt_embeds], dim=0)
            self.byt5_prompt_embeds = torch.cat([self.negative_byt5_prompt_embeds, self.byt5_prompt_embeds], dim=0)
            add_text_embeds = torch.cat([self.negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids[1], add_time_ids[0]], dim=0)

        self.prompt_embeds = self.prompt_embeds.to(device)
        self.byt5_prompt_embeds = self.byt5_prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # if ip_adapter_image is not None or self.loaded_ip_img != ip_adapter_image:
        if (loaded_ip_adapter_img != ip_adapter_image or loaded_clip_vision != image_encoder_type) and image_encoder_type != 'None':
            loaded_clip_vision = image_encoder_type
            print('\033[93m', f'Apply ipadapter by {image_encoder_type} image-feature-extraction.', '\033[0m')
            output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_images_per_prompt, output_hidden_state, image_encoder_type, dtype
            )
            if do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)
            loaded_ip_adapter_img = ip_adapter_image
            loaded_image_embeds = image_embeds
        else:
            image_embeds = loaded_image_embeds

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]
            # timesteps = timesteps[:(num_inference_steps + 1)]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        
        assert batch_size == 1, "batch_size > 1 is not supported"
        
        if loaded_prompt_and_bboxes != [prompt + text_prompt, bboxes]:
            loaded_prompt_and_bboxes = [prompt + text_prompt, bboxes]
            glyph_attn_mask = self.get_glyph_attn_mask(texts, bboxes)
            bg_attn_mask = glyph_attn_mask.sum(-1) == 0
            # 1,h,w,byt5_max_len
            glyph_attn_masks = glyph_attn_mask.unsqueeze(0).to(device)
            # 1,h,w
            bg_attn_masks = bg_attn_mask.unsqueeze(0).to(glyph_attn_masks.dtype).to(device)
            
            # b, h, w, text_feat_len
            glyph_attn_masks = (1 - glyph_attn_masks) * -10000.0
            # b, h, w
            bg_attn_masks = (1 - bg_attn_masks) * -10000.0
            num_down_sample = sum(1 if i == 'CrossAttnDownBlock2D' else 0 for i in self.unet.config['down_block_types']) - 1
            initial_resolution = self.default_sample_size
            initial_resolution = initial_resolution // 2**sum(1 if i == 'DownBlock2D' else 0 for i in self.unet.config['down_block_types'])
            resolution_list = [initial_resolution] + [initial_resolution // 2**i for i in range(1, num_down_sample + 1)]
            glyph_attn_masks_dict = dict()
            bg_attn_masks_dict = dict()
            # b, text_fet_len, h, w
            glyph_attn_masks = glyph_attn_masks.permute(0, 3, 1, 2)
            # b, 1, h, w
            bg_attn_masks = bg_attn_masks.unsqueeze(1)
            for mask_resolution in resolution_list:
                down_scaled_glyph_attn_masks = F.interpolate(
                    glyph_attn_masks, size=(mask_resolution, mask_resolution), mode='nearest',
                )
                # b, text_fet_len, h, w->b, h, w, text_fet_len->b, h*w, text_fet_len
                down_scaled_glyph_attn_masks = down_scaled_glyph_attn_masks.permute(0, 2, 3, 1).flatten(1, 2)
                glyph_attn_masks_dict[mask_resolution * mask_resolution] = down_scaled_glyph_attn_masks

                down_scaled_bg_attn_masks = F.interpolate(
                    bg_attn_masks, size=(mask_resolution, mask_resolution), mode='nearest',
                )
                # b,1,h,w->b,h,w->b,h,w,1->b,h*w,1->b,h*w,clip_feat_len
                down_scaled_bg_attn_masks = down_scaled_bg_attn_masks.squeeze(1).unsqueeze(-1)
                down_scaled_bg_attn_masks = down_scaled_bg_attn_masks.flatten(1, 2)
                down_scaled_bg_attn_masks = down_scaled_bg_attn_masks.repeat(1, 1, self.prompt_embeds.shape[1])
                bg_attn_masks_dict[mask_resolution * mask_resolution] = down_scaled_bg_attn_masks
            if do_classifier_free_guidance:
                for key in glyph_attn_masks_dict:
                    glyph_attn_masks_dict[key] = torch.cat([
                        torch.zeros_like(glyph_attn_masks_dict[key]), 
                        glyph_attn_masks_dict[key]], 
                    dim=0)
                for key in bg_attn_masks_dict:
                    bg_attn_masks_dict[key] = torch.cat([
                        torch.zeros_like(bg_attn_masks_dict[key]), 
                        bg_attn_masks_dict[key]], 
                    dim=0)
            loaded_attn_masks_dicts = [glyph_attn_masks_dict, bg_attn_masks_dict]
        else:
            glyph_attn_masks_dict, bg_attn_masks_dict = loaded_attn_masks_dicts[0], loaded_attn_masks_dicts[1]

        self._num_timesteps = len(timesteps)
        ComfyUI_ProgressBar = ProgressBar(total=num_inference_steps)
        
        if torch.cuda.is_available():
            try:
                self.unet.to(device)
            except torch.cuda.OutOfMemoryError as e: # free vram when OOM
                # self.unet.to('cpu') # 内存大于48GB建议开启，可以在爆显存之后节省显存，内存小于48则不建议，会跑到虚拟内存上。
                print('\033[93m', 'Gpu is out of memory(爆显存了)!', '\033[0m')
                raise e
        else:
            self.unet.to(device)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None and image_encoder_type != 'None':
                    added_cond_kwargs["image_embeds"] = image_embeds
                if self.cross_attention_kwargs is None:
                    cross_attention_kwargs = {}
                else:
                    cross_attention_kwargs = self.cross_attention_kwargs
                cross_attention_kwargs['glyph_encoder_hidden_states'] = self.byt5_prompt_embeds
                cross_attention_kwargs['glyph_attn_masks_dict'] = glyph_attn_masks_dict
                cross_attention_kwargs['bg_attn_masks_dict'] = bg_attn_masks_dict
                
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=self.prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if num_channels_unet == 4:
                    init_latents_proper = image_latents
                    if self.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    self.prompt_embeds = callback_outputs.pop("prompt_embeds", self.prompt_embeds)
                    self.negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", self.negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    self.negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", self.negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    ComfyUI_ProgressBar.update(1)
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
                
            if 'cpu' in self.vae.device.type:
                self.vae.to(device)
                
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            # image = latents
            image = {"samples": latents / 0.13025}

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return (StableDiffusionXLPipelineOutput(images=image), loaded_lora_scale, loaded_prompt_and_bboxes, loaded_embeds, loaded_attn_masks_dicts, loaded_ip_adapter_img, loaded_image_embeds, loaded_clip_vision, )

    def get_glyph_attn_mask(self, texts, bboxes):
        resolution = self.default_sample_size
        text_idx_list = self.get_text_start_pos(texts)
        mask_tensor = torch.zeros(
            resolution, resolution, self.byt5_max_length,
        )
        for idx, bbox in enumerate(bboxes):
            # box is in [x, y, w, h] format
            # area of [y:y+h, x:x+w]
            bbox = [int(v * resolution + 0.5) for v in bbox]
            bbox[2] = max(bbox[2], 1)
            bbox[3] = max(bbox[3], 1)
            bbox[0: 2] =  np.clip(bbox[0: 2], 0, resolution - 1).tolist()
            bbox[2: 4] = np.clip(bbox[2: 4], 1, resolution).tolist()
            mask_tensor[
                bbox[1]: bbox[1] + bbox[3], 
                bbox[0]: bbox[0] + bbox[2], 
                text_idx_list[idx]: text_idx_list[idx + 1]
            ] = 1
        return mask_tensor

    def get_text_start_pos(self, texts):
        prompt = "".encode('utf-8')
        '''
        Text "{text}" in {color}, {type}.
        '''
        pos_list = []
        for text in texts:
            pos_list.append(len(prompt))
            text_prompt = f'Text "{text}"'

            attr_list = ['0', '1']

            attr_suffix = ", ".join(attr_list)
            text_prompt += " in " + attr_suffix
            text_prompt += ". "
            text_prompt = text_prompt.encode('utf-8')

            prompt = prompt + text_prompt
        pos_list.append(len(prompt))
        return pos_list
