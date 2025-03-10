import os
import folder_paths
import re
import cv2
import numpy as np
import torch
from PIL import ImageFont, Image
from comfy.utils import load_torch_file
from ...UL_common.common import pil2tensor, tensor2numpy_cv2, tensor2pil, numpy_cv2tensor, Pillow_Color_Names, clean_up, get_dtype_by_name, CustomModelPatcher, Kwargs, logger
from ... import comfy_temp_dir
from ...UL_common.pretrained_config_dirs import SD15_Base_pretrained_dir
from comfy.model_management import unet_offload_device, get_torch_device, text_encoder_offload_device, total_vram
from comfy.model_patcher import ModelPatcher
import folder_paths
import copy
import latent_preview
from PIL import ImageColor
import comfy.sd

MiaoBi_tokenizer_dir = os.path.join(SD15_Base_pretrained_dir, 'MiaoBi_tokenizer')
Clip_l_tokenizer_dir = os.path.join(SD15_Base_pretrained_dir, 'tokenizer')
Random_Gen_Mask_path = os.path.join(comfy_temp_dir,  "AnyText_random_mask_pos_img.png")
lowvram = total_vram<6000

# is_chinese_prompt = check_chinese(prompt_replace(prompt))

class UL_AnyTextSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AnyText_Model", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "seed": ("INT", {"default": 88888888, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", { "default": 9, "min": 1, "max": 99, "step": 0.1}),
                "strength": ("FLOAT", {"default": 1.00, "min": 0, "max": 2, "step": 0.01}),
                "attnx_scale": ("FLOAT", {"default": 1.00, "min": 0, "max": 2, "step": 0.01}),
                "eta": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.1}),
                # "backend_for_v1": ("BOOLEAN", {"default": True, "label_on": "v2", "label_off": "v1", "tooltip": "", "advanced": False}),
                "keep_load": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Warning: do not delete model unless this node no longer needed, it will try release device_memory and ram. if checked and want to continue node generation, use ComfyUI-Manager `Free model and node cache` to reset node state or change parameter in Loader node to activate.\n注意：仅在这个节点不再需要时删除模型，将尽量释放系统内存和设备专用内存。如果删除后想继续使用此节点，使用ComfyUI-Manager插件的`Free model and node cache`重置节点状态或者更换模型加载节点的参数来激活。"}),
                "keep_device": ("BOOLEAN", {"default": True, "label_on": "comfy", "label_off": "device", "tooltip": "Keep model in comfy_auto_unet_offload_device (HIGH_VRAM: device, Others: cpu) or device_memory after generation.\n生图完成后，模型转移到comfy自动unet选择设备(HIGH_VRAM: device, 其他: cpu)或者保留在设备专用内存上。"}),
            },
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("latent", )
    CATEGORY = "UL Group/Image Generation"
    FUNCTION = "sample"
    TITLE = "AnyText Sampler"
    DESCRIPTION = "AnyText: Multilingual Visual Text Generation And Editing.\nAnyText多语言文字生成与编辑\n通过创新性的算法设计，可以支持中文、英语、日语、韩语等多语言的文字生成，还支持对输入图片中的文字内容进行编辑。本模型所涉及的文字生成技术为电商海报、Logo设计、创意涂鸦、表情包等新型AIGC应用提供了可能性。"

    def sample(self, model, positive, negative, seed, steps, cfg, strength, attnx_scale, eta, keep_load, keep_device, backend_for_v1=True):
        if lowvram:
            model.custom_load(
                [model.model.model, model.model.control_model],
                force_load=True
            )
        else:
            model.load_all_gpu()
        
        isV2 = False
        if model.model_type == "AnyText" and not backend_for_v1:
            from AnyTextControlDiffusion.cldm.ddim_hacked import DDIMSampler
        else:
            from AnyText2ControlDiffusion.cldm.ddim_hacked import DDIMSampler
            isV2 = True
        
        model.model.control_scales = ([strength] * 13)
        if isV2:
            model.model.attnx_scale = attnx_scale
        
        ddim_sampler = DDIMSampler(model.model, device=model.load_device)
        callback = latent_preview.prepare_callback(model, steps) #latent preview
        latents, intermediates = ddim_sampler.sample(
            S=steps, 
            batch_size=positive[0][1]['batch_size'],
            shape=positive[0][1]['shape'], 
            conditioning=positive[0][0], 
            verbose=False, 
            eta=eta,
            unconditional_guidance_scale=cfg,
            unconditional_conditioning=negative[0][0],
            callback=callback, #后端代码有timesteps的callback，已经修改支持latent preview
            generator=torch.Generator(model.load_device).manual_seed(seed),
        )
        
        if keep_load and keep_device and lowvram:
            model.custom_offload(
                [model.model.model, model.model.control_model]
            )
        elif keep_load and keep_device and not lowvram:
            model.offload_or_unload(keep_load, keep_device)
        elif not keep_load:
            model.offload_or_unload(keep_load)
        
        # model['model'].first_stage_model.to(get_torch_device())
        # samples = model['model'].decode_first_stage(latents)
        # samples = (einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        # result = []
        # for sample in samples:
        #     result.append(pil2tensor(sample))
        # result = torch.cat(result, dim=0)
        
        return({"samples": 1. / 0.18215 * latents}, )

class UL_AnyTextLoader:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "control_net_name": (["None"] + folder_paths.get_filename_list("controlnet"), ),
                "miaobi_clip": (["None"] + folder_paths.get_filename_list("text_encoders"), {"advanced": True}),
                "weight_dtype": (["auto", "fp16", "fp32", "bf16", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto", "tooltip": "Only fp16 and fp32 works.\n仅支持fp16和fp32。"}),
                "init_device": (["auto", "device", "cpu"], {"tooltip": "", "advanced": True}),
                # "backend_for_v1": ("BOOLEAN", {"default": True, "label_on": "v2", "label_off": "v1", "tooltip": "", "advanced": True}),
                }
            }

    RETURN_TYPES = ("AnyText_Model", "VAE", "STRING", )
    RETURN_NAMES = ("model", "vae", "ckpt_name", )
    FUNCTION = "Loader"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Loader"
    DESCRIPTION = "Miaobi_clip is optional, for chinese prompt text_encode without translator.\nOption 1: load full AnyText ckeckpoint in ckpt_name without controlnet.\nOption 2: load custom sd1.5 ckeckpoint with AnyText control_net.\nmiaobi_clip是可选项，用于输入中文提示词但不使用翻译机。\n选项1： 加载完整的AnyText模型，此时勿加载control_net。\n选项2：加载自定义sd1.5模型和AnyText的control_net。"
    
    def __init__(self):
        self.model_patcher = None
        self.load_kwargs = None

    def Loader(self, ckpt_name, control_net_name, miaobi_clip, weight_dtype, init_device, backend_for_v1=True):
        import sys
        AnyTextBackbone_Dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Site_Packages")
        sys.path.append(AnyTextBackbone_Dir)
        
        if not backend_for_v1:
            from AnyTextControlDiffusion.cldm.model import create_model
        else:
            from AnyText2ControlDiffusion.cldm.model import create_model
    
        weight_dtype = weight_dtype if weight_dtype in ["fp16", "fp32"] else "fp16"
        dtype = get_dtype_by_name(weight_dtype)
        init_device = unet_offload_device() if init_device == "auto" else torch.device("cpu") if init_device == "cpu" else get_torch_device()
        
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_yaml', 'anytext_sd15.yaml')
        v2_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_yaml', 'anytext2_sd15.yaml')
        is_miaobi, model_type = False, "AnyText"
        
        if self.load_kwargs != (ckpt_name+control_net_name+miaobi_clip+weight_dtype):
            self.load_kwargs = (ckpt_name+control_net_name+miaobi_clip+weight_dtype)
            if self.model_patcher != None:
                self.model_patcher.offload_or_unload(False)
                del self.model_patcher
                
            state_dict = load_torch_file(ckpt_path, safe_load=True)
            if "embedding_manager.font_predictor.backbone.block_list.0._depthwise_conv._batch_norm.num_batches_tracked" in state_dict.keys():
                from AnyText2ControlDiffusion.cldm.model import create_model
                cfg_path, model_type = v2_cfg_path, "AnyText2"
        
            if control_net_name != "None":
                anytext_controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
                anytext_state_dict = load_torch_file(anytext_controlnet_path, safe_load=True)
                if "embedding_manager.font_predictor.backbone.block_list.0._depthwise_conv._batch_norm.num_batches_tracked" in anytext_state_dict.keys():
                    from AnyText2ControlDiffusion.cldm.model import create_model
                    cfg_path, model_type = v2_cfg_path, "AnyText2"
                for k in list(anytext_state_dict):
                    state_dict[k] = anytext_state_dict.pop(k)
                del anytext_state_dict
            
            if miaobi_clip != "None":
                from transformers import AutoTokenizer
                tokenizer =  AutoTokenizer.from_pretrained(MiaoBi_tokenizer_dir, trust_remote_code=True)
                clip_path = folder_paths.get_full_path_or_raise("text_encoders", miaobi_clip)
                for k in list(state_dict.keys()):
                    if k.startswith("cond_stage_model"):
                        state_dict.pop(k)
                clip_l_sd = load_torch_file(clip_path, safe_load=True)
                is_miaobi = True
            else:  
                clip_l_sd = {}
                for k in list(state_dict.keys()):
                    if k.startswith("cond_stage_model"):
                        clip_l_sd[k.replace("cond_stage_model.transformer.", "")] = state_dict.pop(k)
            
            model = create_model(cfg_path, use_fp16=(dtype == torch.float16), is_miaobi=is_miaobi) #dtype control
            
            model.load_state_dict(state_dict, strict=False)
            
            vae_sd = {}
            for k in list(state_dict.keys()):
                if k.startswith("first_stage_model"):
                    vae_sd[k.replace("first_stage_model.", "")] = state_dict.pop(k)
            
            del state_dict
            model.cond_stage_model.transformer.load_state_dict(clip_l_sd, strict=False)
            del clip_l_sd
            if is_miaobi:
                model.cond_stage_model.tokenizer = tokenizer
            model.cond_stage_model.freeze()
            clean_up()
            
            model.eval().to(init_device, dtype)
            
            sub_models = [
                model.first_stage_model,
                model.cond_stage_model,
                model.control_model,
                model.embedding_manager,
                model.text_predictor
            ]
            
            sub_patchers = [
                ModelPatcher(model.cond_stage_model, get_torch_device(), text_encoder_offload_device()),
                ModelPatcher(model.control_model, get_torch_device(), text_encoder_offload_device()),
                ModelPatcher(model.embedding_manager, get_torch_device(), text_encoder_offload_device()),
                ModelPatcher(model.text_predictor, get_torch_device(), text_encoder_offload_device())
            ]
            
            self.model_patcher  =CustomModelPatcher(model, get_torch_device(), unet_offload_device())
            self.model_patcher.sub_models = sub_models
            self.model_patcher.sub_patchers = sub_patchers
            self.model_patcher.dtype = dtype
            self.model_patcher.model_type = model_type
            self.model_patcher.load_kwargs = (ckpt_name+control_net_name+miaobi_clip+weight_dtype)
            self.model_patcher.device = torch.device("cpu")
            self.model_patcher.set_latent_preview()
            
        logger.info(f'image_model_type: {model_type}, current_device: {init_device}, load_device: {self.model_patcher.load_device}, offload_device: {self.model_patcher.offload_device}, compute_dtype: {dtype}')
        
        vae = comfy.sd.VAE(sd=vae_sd)
        del vae_sd
        
        return (self.model_patcher, vae, os.path.basename(ckpt_path), )

class UL_AnyTextFontImg:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "font_name": (["None"] + os.listdir(os.path.join(folder_paths.models_dir, "fonts")), {"default": "None"}),
                "pos_mask": ("MASK", ),
                "sort_radio": ("BOOLEAN", {"default": True, "label_on": "↔水平排序", "label_off": "↕垂直排序"}), 
                "font_color_name": (['transparent'] + Pillow_Color_Names, {"default": "white"}),
                "font_color_code": ("STRING",{"default": "00ffdd"}), 
                "font_color_codeR": ("INT",{"default": -1, "min": -1, "max": 255, "step": 1}), 
                "font_color_codeG": ("INT",{"default": 0, "min": 0, "max": 255, "step": 1}), 
                "font_color_codeB": ("INT",{"default": 0, "min": 0, "max": 255, "step": 1}), 
                "font_color_codeA": ("INT",{"default": 0, "min": 0, "max": 255, "step": 1}), 
                "font_color_mode": ("BOOLEAN", {"default": True, "label_on": "color_name", "label_off": "color_code"}), 
                "bg_color_name": (['transparent'] + Pillow_Color_Names, {"default": "transparent"}),
                "bg_color_code": ("STRING",{"default": "00ffdd"}), 
                "bg_color_mode": ("BOOLEAN", {"default": True, "label_on": "color_name", "label_off": "color_code"}),
                "seperate_by": ("STRING",{"default": "---"}),  
                "prompt": ("STRING", {"default": "你好呀---Hello!", "multiline": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                }
            }

    RETURN_TYPES = ("IMAGE", )#"STRING", )
    RETURN_NAMES = ("font_img", )#"merged_dir", )
    FUNCTION = "FontImg"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText FontImg"

    def FontImg(self, font_name, pos_mask, prompt, width, height, sort_radio, font_color_name, font_color_code, font_color_mode, bg_color_name, bg_color_code, bg_color_mode, font_color_codeR, font_color_codeG, font_color_codeB, font_color_codeA, seperate_by):
        texts = str(prompt).split(seperate_by)
        n_lines = len(texts)
        if len(texts ) == 0:
            texts  = [' ']
        max_chars = 50
        font_path = os.path.join(folder_paths.models_dir, "fonts", font_name)
        
        if not os.path.isfile(font_path):
            raise ValueError("Invalid font path.\n无效字体路径。")
        
        font = ImageFont.truetype(font_path, size=60, encoding='utf-8')
        mask_img = tensor2numpy_cv2(pos_mask)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB) # cv2二值图(mask)转rgb
        mask_img = cv2.bitwise_not(mask_img) # cv2图片取反
        
        if font_color_mode:
            font_color = font_color_name
            if font_color_name == 'transparent':
                font_color = (0,0,0,0)
            # if font_color_name == 'transparent' or font_color_name == 'black':
            #     raise ValueError('黑色和透明图字体在本脚本暂时无法生成！')
        elif not font_color_mode and font_color_codeR == -1:
            font_color = "#" + str(font_color_code).replace("#", "").replace(":", "").replace(" ", "")
            # if '000000' in font_color:
            #     raise ValueError('黑色和透明图字体在本脚本暂时无法生成！')
        else:
            font_color = (font_color_codeR, font_color_codeG, font_color_codeB, font_color_codeA)
            
        if bg_color_mode:
            bg_color = bg_color_name
            if bg_color_name == 'transparent':
                bg_color = (0,0,0,0)
        else:
            bg_color =  "#" + str(bg_color_code).replace("#", "").replace(":", "").replace(" ", "") # 颜色码-绿色：#00FF00
        
        from .AnyText_scripts.AnyText_pipeline_util import resize_image
        from .AnyText_scripts.AnyText_t3_dataset import draw_glyph2
        from .AnyText_scripts.AnyText_pipeline import separate_pos_imgs, find_polygon
        pos_image = resize_image(mask_img, max_length=768)
        pos_image = cv2.resize(pos_image, (width, height))
        pos_imgs = 255-pos_image
        pos_imgs = pos_imgs[..., 0:1]
        pos_imgs = cv2.convertScaleAbs(pos_imgs)
        _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
        
        if sort_radio:
            sort_radio = '↔'
        else:
            sort_radio = '↕'
        
        pos_imgs = separate_pos_imgs(pos_imgs, sort_radio)
        if len(pos_imgs) == 0:
            pos_imgs = [np.zeros((height, width, 1))]
        if len(pos_imgs) < n_lines:
            if n_lines == 1 and texts[0] == ' ':
                # pass  # text-to-image without text
                print('\033[93m', f'Warning: text-to-image without text.', '\033[0m')
            else:
                raise ValueError(f'Found {len(pos_imgs)} positions that < needed {n_lines} from prompt, check and try again(手绘遮罩数少于要绘制的文本数，检查再重试)!')
        elif len(pos_imgs) > n_lines:
            print('\033[93m', f'Warning: found {len(pos_imgs)} positions that > needed {n_lines} from prompt.', '\033[0m')
        pre_pos = []
        poly_list = []
        
        for input_pos in pos_imgs:
            if input_pos.mean() != 0:
                input_pos = input_pos[..., np.newaxis] if len(input_pos.shape) == 2 else input_pos
                poly, pos_img = find_polygon(input_pos)
                pre_pos += [pos_img/255.]
                poly_list += [poly]
            else:
                pre_pos += [np.zeros((height, width, 1))]
                poly_list += [None]
        
        board = Image.new('RGBA', (width, height), bg_color) # 最终图的底图，颜色任意，只是个黑板。
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > max_chars:
                text = text[:max_chars]
            gly_scale = 2
            if pre_pos[i].mean() != 0:
                _, glyphs = draw_glyph2(font, text, poly_list[i], scale=gly_scale, width=width, height=height, add_space=False, font_color=font_color)
            glyphs = glyphs.convert('RGBA')
            glyphs = glyphs.resize(size=(board.width, board.height)) # 缩放字体图以匹配输入尺寸
            r,g,b,a = glyphs.split() # 读取字体图片中透明像素
            board.paste(glyphs, (0,0), mask=a) # 使用字体图alpha通道像素作遮罩，将非alpha的字体图粘贴到board。
            
        font_img = pil2tensor(board)
        
        return (font_img, )
        
class UL_AnyTextComposer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": ("BOOLEAN", {"default": True, "label_on": "cv2_add", "label_off": "pil_paste", "tooltip": "cv2_add for canny img, pil_paste for img with alpha channel."}),
                "font_or_bg_img": ("IMAGE", ),
                }, 
                "optional": {
                "font_img2": ("IMAGE", ),
                "font_img3": ("IMAGE", ),
                "font_img4": ("IMAGE", ),
                "font_img5": ("IMAGE", ),
                "font_img6": ("IMAGE", ),
                "font_img7": ("IMAGE", ),
                "font_img8": ("IMAGE", ),
                "font_img9": ("IMAGE", ),
                "font_img10": ("IMAGE", ),
                }
            }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("font_img", )
    FUNCTION = "composer"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Composer"
    
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,)

    def composer(self, mode, font_or_bg_img=None, font_img2=None, font_img3=None, font_img4=None, font_img5=None, font_img6=None, font_img7=None, font_img8=None, font_img9=None, font_img10=None):
        images = [font_img2, font_img3, font_img4, font_img5, font_img6, font_img7, font_img8, font_img9, font_img10]
        # new_font_img = font_img1
        # for img in images:
        #     if img != None:
        #         new_font_img += img
        if mode:
            # new_font_img = None
            new_font_img = tensor2numpy_cv2(font_or_bg_img)
            for img in images:
                if img != None:
                    img = tensor2numpy_cv2(img)
                    new_font_img += img
            new_font_img = numpy_cv2tensor(new_font_img)
        else:
            bg_img = tensor2pil(font_or_bg_img)
            # new_font_img = Image.new('RGBA', size=bg_img.size, color=(0,0,0,0))
            bg_img = bg_img.convert('RGBA')
            font_or_bg_img = pil2tensor(bg_img)
            for img in images:
                if img != None:
                    img = tensor2pil(img)
                    img = img.convert('RGBA')
                    r,g,b,a = img.split()
                    bg_img.paste(img, (0,0), mask=a)
            new_font_img = pil2tensor(bg_img)
        
        return (new_font_img, )
    
class UL_AnyTextEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        font_list = os.listdir(os.path.join(folder_paths.models_dir, "fonts"))
        return {
            "required": {
                "model": ("AnyText_Model", ),
                "mask": ("MASK", ),
                "prompt": ("STRING", {"forceInput": True}),
                "texts": ("LIST", ),
                "latent": ("LATENT", ),
                "font_name": (['Auto_DownLoad'] + font_list, {"default": "AnyText-Arial-Unicode.ttf"}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "text-generation生成", "label_off": "text-editing文字编辑"}),
                "sort_radio": ("BOOLEAN", {"default": True, "label_on": "↔水平", "label_off": "↕垂直", "tooltip": "Order of draw texts according to mask position orders. ↕ for y axis. It will draw text-content(“string”) from start-to-end(order) on the mask position from top to bottom. ↔ for x axis .It will draw text-content(“string”) from start-to-end(order) on the mask position from left to right.\n根据遮罩位置顺序决定生成文本的顺序。"}),
                "a_prompt": ("STRING", {"default": "best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks", "multiline": True}),
                "n_prompt": ("STRING", {"default": "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture", "multiline": True}),
                "random_mask": ("BOOLEAN", {"default": False, "tooltip": "Random generate mask, the input mask will be ignored.\n随机生成遮罩，输入的遮罩将被忽略。"}),
                "revise_pos": ("BOOLEAN", {"default": False, "tooltip": "Which uses the bounding box of the rendered text as the revised position. However, it is occasionally found that the creativity of the generated text is slightly lower using this method, It dosen’t work in text-edit mode.\n使用边界盒子渲染文字作位置调整。但是发现偶尔会影响生成质量，仅在使用随机生成遮罩时生效。"}),
            },
            "optional": {
                "image": ("IMAGE", ),
                "fonts": ("ANYTEXT_FONTS", ),
                "font_apply": ("BOOLEAN", {"default": True, "label_on": "official", "label_off": "custom", "tooltip": "", "advanced": True}),
                "show_glyph": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "", "advanced": True}),
            }
        }

    # RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK", )
    # RETURN_NAMES = ("positive", "negative", "mask_img", "masked_x", "font_img", "gly_line", "glyphs", "masked_img", "font_hint_img", "mask", )
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE", "IMAGE", )
    RETURN_NAMES = ("positive", "negative", "font_hint", "font", )
    FUNCTION = "encoder"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Encoder"
    
    def __init__(self):
        self.font = None
        self.font_name = None
        self.loaded = False
        self.load_kwargs = None
    
    def encoder(self, model, font_name, mask, prompt, texts, latent, mode, sort_radio, a_prompt, n_prompt, revise_pos, random_mask, image=None, fonts=None, show_glyph=False, font_apply=True):
        if not self.loaded or self.loaded != model.load_kwargs or not lowvram:
            self.loaded = True
            self.load_kwargs = model.load_kwargs
            model.load_all_gpu()
            
        if lowvram:
            model.custom_load(
                [
                    model.model.cond_stage_model,
                    model.model.first_stage_model,
                    model.model.embedding_manager,
                    model.model.text_predictor
                ]
            )
        
        font_path = os.path.join(folder_paths.models_dir, "fonts", font_name)
        font_path = os.path.join(folder_paths.models_dir, "fonts", "SourceHanSansSC-Medium.otf") if not os.path.exists(font_path) else font_path
        if not os.path.exists(font_path):
            from huggingface_hub import hf_hub_download as hg_hf_hub_download
            hg_hf_hub_download(
                repo_id="Sanster/AnyText", 
                filename="SourceHanSansSC-Medium.otf", 
                local_dir=os.path.join(folder_paths.models_dir, "fonts"), 
                )
                
        if self.font_name != font_name: #Avoid duplicate font load.
            self.font = ImageFont.truetype(font_path, size=60, encoding='utf-8')
                    
        if model.model_type == "AnyText":
            from .AnyText_scripts.AnyText_Infer import encode
        else:
            from .AnyText_scripts.AnyText2_Infer import encode
            
        cond, un_cond, h, w, font_hint_img, glyph_img = encode(
            model=model,
            font=self.font,
            mask=mask,
            prompt=prompt,
            texts=texts,
            latent=latent,
            mode=mode,
            sort_radio=sort_radio,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            revise_pos=revise_pos,
            random_mask=random_mask,
            Random_Gen_Mask_path=Random_Gen_Mask_path,
            image=image,
            fonts=fonts,
            font_apply=font_apply,
            show_glyph=show_glyph
        )
        
        if lowvram:
            model.custom_offload(
                [
                    model.model.cond_stage_model,
                    model.model.first_stage_model,
                    model.model.embedding_manager,
                    model.model.text_predictor
                ],
                do_clean=True
            )
        
        if show_glyph:
            font_hint_list = []
            glyph_dir = os.path.join(comfy_temp_dir, "AnyText")
            for i in os.listdir(glyph_dir):
                if i.startswith("glyph2_"):
                    img = pil2tensor(Image.open(os.path.join(glyph_dir, i)))
                    font_hint_list.append(img)
            font_hint_img = torch.cat(font_hint_list, dim=0)
        
        # return ([[cond, {"pooled_output": {}, "shape": (4, h // 8, w // 8), "batch_size": latent["samples"].shape[0]}]], [[un_cond, {"pooled_output": {}}]], mask, {"samples": 1. / 0.18215 * masked_x}, font_img, gly_line_img, glyphs_img, masked_img, font_hint_img, glyph_img, )
        return ([[cond, {"pooled_output": {}, "shape": (4, h // 8, w // 8), "batch_size": latent["samples"].shape[0]}]], [[un_cond, {"pooled_output": {}}]], font_hint_img, glyph_img, )
    
class UL_AnyTextFormatter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": 'close-up of hakurei reimu sitting in a room, with text: "博丽灵梦" on the wall.\nclose-up of 18yo beautiful chinese girl, standing in a county yard, with text: "emm" and "博丽灵梦" in the background.', "multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                
            }
        }

    RETURN_TYPES = ("STRING", "LIST", )
    RETURN_NAMES = ("prompt", "texts", )
    FUNCTION = "formatter"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Formatter"
    
    def formatter(self, prompt):
        prompt, texts = modify_prompt(
            prompt=prompt,
        )
        return (prompt, texts)
    
class UL_AnyText2Fonts:
    @classmethod
    def INPUT_TYPES(cls):
        font_list = ['None'] + os.listdir(os.path.join(folder_paths.models_dir, "fonts"))
        AnyText_colors = copy.deepcopy(["None"]+Pillow_Color_Names)
        return {
            "required": {
                "font_hollow": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "", "advanced": False}),
                "font_name": (font_list, {"default": "AnyText-Arial-Unicode.ttf"}),
                "font_color": (AnyText_colors, {"default": "red"}),
                
                "font_name1": (font_list, {"default": "阿里妈妈东方大楷.otf"}),
                "font_color1": (AnyText_colors, {"default": "green"}),
                
                "font_name2": (font_list, {"default": "仿乾隆字体.ttf"}),
                "font_color2": (AnyText_colors, {"default": "blue"}),
                
                "font_name3": (font_list, {"default": "站酷小薇LOGO体.otf"}),
                "font_color3": (AnyText_colors, {"default": "white"}),
                
                "font_name4": (font_list, {"default": "None"}),
                "font_color4": (AnyText_colors, {"default": "None"}),
                
                "font_name5": (font_list, {"default": "索尼兰亭.ttf"}),
                "font_color5": (AnyText_colors, {"default": "yellow"}),
                
                "font_name6": (font_list, {"default": "日系筑紫a丸GBK版.ttf"}),
                "font_color6": (AnyText_colors, {"default": "pink"}),
                
                "font_name7": (font_list, {"default": "阿朱泡泡体.ttf"}),
                "font_color7": (AnyText_colors, {"default": "gold"}),
            },
            "optional": {
                # "image": ("IMAGE", ),
                # "image1": ("IMAGE", ),
                # "image2": ("IMAGE", ),
                # "image3": ("IMAGE", ),
                # "image4": ("IMAGE", ),
                # "image5": ("IMAGE", ),
                # "image6": ("IMAGE", ),
                # "image7": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("ANYTEXT_FONTS", "DICT", )
    RETURN_NAMES = ("fonts", "kwargs", )
    FUNCTION = "main"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText2 Fonts"
    
    def main(self, font_hollow, **font_kwargs):
        try:
            import webcolors
        except Exception as e:
            raise ValueError("webcolors not installed, python pip install webcolors")
        
        font_paths = []
        font_colors = []
                
        for i, j in font_kwargs.items():
            if "font_name" in i:
                font_path = os.path.join(folder_paths.models_dir, "fonts", j)
                if os.path.exists(font_path):
                    font_paths.append(ImageFont.truetype(font_path, size=60))
                else:
                    font_paths.append('No Font(不指定字体)')
            if "font_color" in i:
                if j != "None":
                    font_colors.append(f'rgba{webcolors.name_to_rgb(j)+(1,)}')
                else:
                    font_colors.append(None)
                    
        
        glyline_font_path = ['None'] * 8
        text_colors = ' '.join(['500,500,500']*8)
        for idx, f in enumerate(font_paths):
            if f == 'No Font(不指定字体)':
                pass
            # elif f == 'Mimic From Image(模仿图中字体)':
            #     img = mimic_list[idx]
            #     if 'layers' in img and img['layers'][0][..., 3:].mean() > 0:
            #         font_hint_image[idx] = img['background'][..., :3][..., ::-1]
            #         font_hint_mask[idx] = img['layers'][0][..., 3:]
            #     else:
            #         font_hint_image[idx] = None
            #         font_hint_mask[idx] = None
            else:
                glyline_font_path[idx] = f
        for idx, c in enumerate(font_colors):
            if c is not None:
                strs = text_colors.split()
                if isinstance(c, str) and 'rgba' in c:
                    rgb = [int(float(i)) for i in c.split('(')[-1].split(')')[0].split(',')[:3]]  # for gradio 5.X
                else:
                    rgb = ImageColor.getcolor(c, "RGB")
                if list(rgb) == [0, 0, 0] or rgb == [255, 255, 255]:
                    rgb = (500, 500, 500)
                rgb = ','.join([str(i) for i in list(rgb)])
                strs[idx] = rgb
                text_colors = ' '.join(strs)
                
        kwargs_ori = {
            "glyline_font_path": glyline_font_path,
            "text_colors": text_colors,
            "font_hollow": font_hollow
        }
        kwargs = Kwargs(kwargs_ori)
        
        return (kwargs, kwargs_ori, )

# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "UL_AnyText_Sampler": UL_AnyTextSampler,
    "UL_AnyText_Loader": UL_AnyTextLoader,
    "UL_AnyText_FontImg": UL_AnyTextFontImg,
    "UL_AnyText_Composer": UL_AnyTextComposer,
    "UL_AnyTextEncoder": UL_AnyTextEncoder,
    "UL_AnyTextFormatter": UL_AnyTextFormatter,
    "UL_AnyText2Fonts": UL_AnyText2Fonts
}

def prompt_replace(prompt):
    #将中文符号“”中的所有内容替换为空内容，防止输入中文被检测到，从而加载翻译模型。
    # prompt = replace_between(prompt, "“", "”", "*")
    prompt = prompt.replace('“', '"').replace('”', '"')
    p = '"(.*?)"'
    strs = re.findall(p, prompt)
    if len(strs) == 0:
        strs = [' ']
    else:
        for s in strs:
            prompt = prompt.replace(f'"{s}"', f'*', 1)
    return prompt

# def replace_between(s, start, end, replacement):
#     # 正则表达式，用以匹配从start到end之间的所有字符
#     pattern = r"%s(.*?)%s" % (re.escape(start), re.escape(end))
#     # 使用re.DOTALL标志来匹配包括换行在内的所有字符
#     return re.sub(pattern, replacement, s, flags=re.DOTALL)

# def check_chinese(text):
#     from .AnyText_scripts.AnyText_bert_tokenizer import BasicTokenizer
#     checker = BasicTokenizer()
#     text = checker._clean_text(text)
#     for char in text:
#         cp = ord(char)
#         if checker._is_chinese_char(cp):
#             return True
#     return False

def modify_prompt(prompt):
    old_prompt = prompt
    PLACE_HOLDER = '*'
    prompt = prompt.replace('“', '"')
    prompt = prompt.replace('”', '"')
    p = '"(.*?)"'
    strs = re.findall(p, prompt)
    if len(strs) == 0:
        strs = [' ']
    else:
        for s in strs:
            prompt = prompt.replace(f'"{s}"', f' {PLACE_HOLDER} ', 1)
    print(f'\033[93mFormat prompt: {old_prompt} --> {prompt}\033[0m')
    return prompt, strs