import os
import folder_paths
from .common import get_dtype_by_name
from comfy.utils import load_torch_file

class UL_Common_Diffusers_Checkpoint_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (['None'] + folder_paths.get_filename_list("checkpoints"), {"tooltip": 'Checkpoints, sd1.5、sdxl (including inpaint model) and sd2.1.\n基座模型，sd1.5、sdxl(包括重绘模型)和sd2.1。'}),
                "unet_only": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "Try decrease resource comsuption.\n尝试减少资源占用。"}),
                "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto"}),
                }
            }

    RETURN_TYPES = ("Diffusers_Model", "STRING", )
    RETURN_NAMES = ("diffusers_model", "ckpt_name", )
    FUNCTION = "UL_Common_Diffusers_Checkpoint_Loader"
    CATEGORY = "UL Group/Common Loader"
    TITLE = "Diffusers Load Checkpoint"
    DESCRIPTION = "Use diffusers library to load single_file checkpoint, now support sd1.5、sd2.1 and sdxl (including cosxl).\n使用diffusers库加载单文件模型，现在支持sd1.5、sd2.1和sdxl(包括cosxl)。"

    def UL_Common_Diffusers_Checkpoint_Loader(self, ckpt_name, dtype, debug=False, unet_only=False):
        
        dtype = get_dtype_by_name(dtype)
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        
        state_dict = load_torch_file(ckpt_path)
        
        # text = ''
        out_blocks = 0 # sd1.5、sd2.1: 384 sdxl、cosxl: 868 sdxl-flash-mini: 588 hunyuan_dit_1.2: 192.
        for param_name, param in state_dict.items():
            if 'output' in param_name:
                out_blocks += 1
            # text += f'{param_name}:\n{param.shape}\n'
        # with open(f'C:/Users/pc/Desktop/sd_{os.path.basename(ckpt_path)}.txt', 'w', encoding='utf-8') as f:
        #     f.write(text)
            # f.write(str(state_dict))
            
        if debug:
            print('\033[93m', f'Out blocks: {out_blocks}', '\033[0m')
        
        if out_blocks == 384: # sd1.5 or sd2.1
            unet_num_in_channels = state_dict['model.diffusion_model.input_blocks.0.0.weight'].shape[1] # torch.Size, cosxl: torch.Size([320, 8, 3, 3]) sd1.5、sd2.1、sdxl base: torch.Size([320, 4, 3, 3]) inpaint: torch.Size([320, 9, 3, 3]).
            from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
            from .pretrained_config_dirs import SD15_Base_pretrained_dir, SD15_Inpaint_Base_pretrained_dir, SD21_Base_pretrained_dir
            
            config_dir = SD15_Base_pretrained_dir
            original_config_path = os.path.join(folder_paths.get_full_path_or_raise('configs', 'v1-inference_fp16.yaml'))
            pipe = StableDiffusionPipeline
            
            if unet_num_in_channels == 9: # inpaint
                config_dir = SD15_Inpaint_Base_pretrained_dir
                original_config_path = folder_paths.get_full_path_or_raise('configs', 'v1-inpainting-inference.yaml')
                pipe = StableDiffusionInpaintPipeline
            elif 'cond_stage_model.model.ln_final.bias' in state_dict.keys(): # sd2.1 
                config_dir = SD21_Base_pretrained_dir
                original_config_path = folder_paths.get_full_path_or_raise('configs', 'v2-inference.yaml')
            
            if unet_only:
                if debug:
                    if 'cond_stage_model.model.ln_final.bias' in state_dict.keys(): # sd2.1
                        print('\033[93m', '`cond_stage_model.model.ln_final.bias` in state_dict and out_blocks: 384, load sd2.1 unet.', '\033[0m')
                    elif unet_num_in_channels == 9:
                        print('\033[93m', 'Unet in_channels: 9 and out_blocks: 384, load sd1.5 inpainting unet.', '\033[0m')
                    else:
                        print('\033[93m', 'Out_blocks: 384, load sd1.5 unet.', '\033[0m')
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
                    if unet_num_in_channels == 9:
                        print('\033[93m', 'Unet in_channels: 9 and out_blocks: 384, load sd1.5 inpainting checkpoint.', '\033[0m')
                    elif 'cond_stage_model.model.ln_final.bias' in state_dict.keys():
                        print('\033[93m', '`cond_stage_model.model.ln_final.bias` in state_dict and out_blocks: 384, load sd2.1 checkpoint.', '\033[0m')
                    else:
                        print('\033[93m', 'Out_blocks: 384, load sd1.5 checkpoint.', '\033[0m')
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
        elif out_blocks == 868: # sdxl
            unet_num_in_channels = state_dict['model.diffusion_model.input_blocks.0.0.weight'].shape[1]
            from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline
            from .pretrained_config_dirs import SDXL_Base_pretrained_dir, SDXL_Inpaint_Base_pretrained_dir
            
            config_dir = SDXL_Base_pretrained_dir
            pipe = StableDiffusionXLPipeline
            
            original_config_path = os.path.join(folder_paths.get_full_path_or_raise('configs', 'sd_xl_base.yaml'))
            
            if unet_num_in_channels == 8:
                original_config_path = os.path.join(folder_paths.get_full_path_or_raise('configs', 'sd_xl_cosxl_base.yaml'))
            
            if unet_num_in_channels == 9: # inpaint
                config_dir = SDXL_Inpaint_Base_pretrained_dir
                pipe = StableDiffusionXLInpaintPipeline
                original_config_path=os.path.join(folder_paths.get_full_path_or_raise('configs', 'sd_xl-inpainting_base.yaml'))
                
            if unet_only:
                if debug:
                    if unet_num_in_channels == 9:
                        print('\033[93m', 'Unet in_channels: 9 and out_blocks: 868, load sdxl inpainting unet.', '\033[0m')
                    elif unet_num_in_channels == 8:
                        print('\033[93m', 'Unet in_channels: 8 and out_blocks: 868, load sdxl consxl unet.', '\033[0m')
                    else:
                        print('\033[93m', 'Out_blocks: 868, load sdxl unet.', '\033[0m')
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
                    if unet_num_in_channels == 9:
                        print('\033[93m', 'Unet in_channels: 9 and out_blocks: 868, load sdxl inpainting checkpoint.', '\033[0m')
                    elif unet_num_in_channels == 8:
                        print('\033[93m', 'Unet in_channels: 8 and out_blocks: 868, load sdxl cosxl checkpoint.', '\033[0m')
                    else:
                        print('\033[93m', 'Out_blocks: 868, load sdxl checkpoint.', '\033[0m')
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
        elif out_blocks == 588:
            raise ValueError(f'sdxl-flash-mini checkpoint not supported in diffusers library.')
        elif out_blocks == 192:
            raise ValueError(f'HunyuanDit not supported.')
        elif out_blocks == 795:
            raise ValueError(f'AnimateLCM-SVD-xt-1-1 not supported.')
        else:
            raise ValueError(f'Can not detect model type.')
        
        del state_dict
        
        model = {
            'pipe': pipeline, 
            'unet': unet, 
            'clip': clip, 
            'vae': vae, 
            'scheduler': scheduler,
        }
        
        return (model, os.path.basename(ckpt_path), )
    
NODE_CLASS_MAPPINGS = {
    "UL_Common_Diffusers_Checkpoint_Loader": UL_Common_Diffusers_Checkpoint_Loader,
}