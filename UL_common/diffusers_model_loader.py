import os
import folder_paths
from .common import get_dtype_by_name

class UL_Common_Diffusers_Checkpoint_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (['None'] + folder_paths.get_filename_list("checkpoints"), {"tooltip": 'Checkpoints, now for sd1.5 and sdxl (including inpaint model). For inpaint model, model name must contain inpaint.'}),
                "unet_only": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "Decrease resource comsuption."}),
                "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto"}),
                }
            }

    RETURN_TYPES = ("Diffusers_Model", "STRING", )
    RETURN_NAMES = ("diffusers_model", "ckpt_name", )
    FUNCTION = "UL_Common_Diffusers_Checkpoint_Loader"
    CATEGORY = "UL Group/Common Loader"
    TITLE = "Diffusers Load Checkpoint"
    DESCRIPTION = "Use diffusers library to load single_file checkpoint, now support sd1.5、sd2.1 and sdxl, recognize model type by size (1.5GB < sd1.5 & sd2.1 < 6GB, 6GB < sdxl < 7GB)、inapint (inpaint in ckpt_name) and sd2.1 (sd21 in ckpt_name)."

    def UL_Common_Diffusers_Checkpoint_Loader(self, ckpt_name, dtype, debug=True, unet_only=False):
        
        dtype = get_dtype_by_name(dtype)
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        
        if 1610612736 < (os.stat(ckpt_path)).st_size < 6442450944: # 1.5GB<ckpt_size<6GB, sd1.5
            from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
            from .pretrained_config_dirs import SD15_Base_pretrained_dir, SD15_Inpaint_Base_pretrained_dir, SD21_Base_pretrained_dir
            
            config_dir = SD15_Base_pretrained_dir
            original_config_path = os.path.join(folder_paths.get_full_path_or_raise('configs', 'v1-inference_fp16.yaml'))
            pipe = StableDiffusionPipeline
            
            if 'inpaint' in ckpt_path: # inpaint
                config_dir = SD15_Inpaint_Base_pretrained_dir
                original_config_path = folder_paths.get_full_path_or_raise('configs', 'v1-inpainting-inference.yaml')
                pipe = StableDiffusionInpaintPipeline
            elif 'sd21' in ckpt_path: # sd2.1
                config_dir = SD21_Base_pretrained_dir
                original_config_path = folder_paths.get_full_path_or_raise('configs', 'v2-inference.yaml')
            
            if unet_only:
                if debug:
                    if 'sd21' in ckpt_path: # sd2.1
                        print('\033[93m', 'Load sd2.1 unet.', '\033[0m')
                    elif 'inpaint' in ckpt_path:
                        print('\033[93m', 'Load sd2.1 inpainting unet.', '\033[0m')
                    else:
                        print('\033[93m', 'Load sd1.5 unet.', '\033[0m')
                from diffusers import UNet2DConditionModel
                unet = UNet2DConditionModel.from_single_file(
                    pretrained_model_link_or_path_or_dict=ckpt_path,
                    original_config=original_config_path,
                    torch_dtype=dtype,
                )
                pipeline = None
                clip = None
                vae = None
                scheduler = None
            else:
                if debug:
                    if 'inpaint' in ckpt_path:
                        print('\033[93m', '`inpaint` in checkpoint name, load sd1.5 inpainting checkpoint.', '\033[0m')
                    elif 'sd21' in ckpt_path:
                        print('\033[93m', '`sd21` in checkpoint name, load sd2.1 checkpoint.', '\033[0m')
                    else:
                        print('\033[93m', 'Load sd1.5 checkpoint.', '\033[0m')
                pipeline = pipe.from_single_file(
                    pretrained_model_link_or_path=ckpt_path,
                    config=config_dir, 
                    original_config=original_config_path,
                    requires_safety_checker = False, 
                    safety_checker=None,
                    torch_dtype=dtype,
                )
                pipeline.tokenizer_2 = None
                pipeline.text_encoder_2 = None
                unet = pipeline.unet
                vae = pipeline.vae
                scheduler = pipeline.scheduler
                clip = {
                    "tokenizer": pipeline.tokenizer,
                    "tokenizer_2": pipeline.tokenizer_2,
                    "text_encoder": pipeline.text_encoder,
                    "text_encoder_2": pipeline.text_encoder_2,
                }
        elif 6442450944 < (os.stat(ckpt_path)).st_size < 7516192768: # 6GB<ckpt_size<7GB, sdxl
            from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline
            from .pretrained_config_dirs import SDXL_Base_pretrained_dir, SDXL_Inpaint_Base_pretrained_dir
            
            config_dir = SDXL_Base_pretrained_dir
            pipe = StableDiffusionXLPipeline
            
            original_config_path = os.path.join(folder_paths.get_full_path_or_raise('configs', 'sd_xl_base.yaml'))
            if 'inpaint' in ckpt_path: # inpaint
                config_dir = SDXL_Inpaint_Base_pretrained_dir
                pipe = StableDiffusionXLInpaintPipeline
                original_config_path=os.path.join(folder_paths.get_full_path_or_raise('configs', 'sd_xl-inpainting_base.yaml'))
                
            if unet_only:
                if debug:
                    if 'inpaint' in ckpt_path:
                        print('\033[93m', 'Load sdxl inpainting unet.', '\033[0m')
                    else:
                        print('\033[93m', 'Load sdxl unet.', '\033[0m')
                from diffusers import UNet2DConditionModel
                unet = UNet2DConditionModel.from_single_file(
                    pretrained_model_link_or_path_or_dict=ckpt_path,
                    original_config=original_config_path,
                    torch_dtype=dtype,
                )
                pipeline = None
                clip = None
                vae = None
                scheduler = None
            else:
                if debug:
                    if 'inpaint' in ckpt_path:
                        print('\033[93m', '`inpaint` in checkpoint name, load sdxl inpainting checkpoint.', '\033[0m')
                    else:
                        print('\033[93m', 'Load sdxl checkpoint.', '\033[0m')
                pipeline = pipe.from_single_file(
                    pretrained_model_link_or_path=ckpt_path,
                    config=config_dir,
                    original_config=original_config_path,
                    torch_dtype=dtype,
                )
                unet = pipeline.unet
                vae = pipeline.vae
                scheduler = pipeline.scheduler
                clip = {
                    "tokenizer": pipeline.tokenizer,
                    "tokenizer_2": pipeline.tokenizer_2,
                    "text_encoder": pipeline.text_encoder,
                    "text_encoder_2": pipeline.text_encoder_2,
                }
        elif 'flash-mini' in ckpt_path or 'Flash_Mini' in ckpt_path:
            raise ValueError(f'sdxl-flash-mini checkpoint not supported in diffusers.')
        else:
            raise ValueError(f'Can not recognize model type.')
        
        model = {
            'pipe': pipeline, 
            'unet': unet, 
            'clip': clip, 
            'vae': vae, 
            'scheduler': scheduler,
        }
        
        return (model, ckpt_name, )
    
NODE_CLASS_MAPPINGS = {
    "UL_Common_Diffusers_Checkpoint_Loader": UL_Common_Diffusers_Checkpoint_Loader,
}