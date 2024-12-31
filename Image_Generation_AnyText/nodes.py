import os
import folder_paths
import re
import cv2
import numpy as np
import torch
from PIL import ImageFont, Image
from comfy.utils import load_torch_file, ProgressBar
from .AnyText_scripts.AnyText_pipeline_util import resize_image
from ..UL_common.common import pil2tensor, get_device_by_name, tensor2numpy_cv2, download_repoid_model_from_huggingface, tensor2pil, numpy_cv2tensor, Pillow_Color_Names, clean_up, get_dtype_by_name, comfy_clean_vram
from .. import comfy_temp_dir
from ..UL_common.pretrained_config_dirs import SD15_Base_pretrained_dir
from comfy.model_management import unet_offload_device, get_torch_device, text_encoder_offload_device, soft_empty_cache, vae_offload_device, load_model_gpu
import folder_paths
import einops
import copy

MiaoBi_tokenizer_dir = os.path.join(SD15_Base_pretrained_dir, 'MiaoBi_tokenizer')
Random_Gen_Mask_path = os.path.join(comfy_temp_dir,  "AnyText_random_mask_pos_img.png")

class UL_AnyTextSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AnyText_Model", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ), 
                "seed": ("INT", {"default": 88888888, "min": 0, "max": 4294967296}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", { "default": 9, "min": 1, "max": 99, "step": 0.1}),
                "strength": ("FLOAT", {"default": 1.00, "min": 0, "max": 2, "step": 0.01}),
                "eta": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.1}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Warning: do not delete model unless this node no longer needed, it will try release device_memory and ram. if checked and want to continue node generation, use ComfyUI-Manager `Free model and node cache` to reset node state or change parameter in Loader node to activate.\n注意：仅在这个节点不再需要时删除模型，将尽量释放系统内存和设备专用内存。如果删除后想继续使用此节点，使用ComfyUI-Manager插件的`Free model and node cache`重置节点状态或者更换模型加载节点的参数来激活。"}),
                "keep_model_device": ("BOOLEAN", {"default": True, "label_on": "comfy", "label_off": "device", "tooltip": "Keep model in comfy_auto_unet_offload_device (HIGH_VRAM: device, Others: cpu) or device_memory after generation.\n生图完成后，模型转移到comfy自动unet选择设备(HIGH_VRAM: device, 其他: cpu)或者保留在设备专用内存上。"}),
            },
        }

    # RETURN_TYPES = ("IMAGE", "LATENT", )
    # RETURN_NAMES = ("image", "latent", )
    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("latent", )
    CATEGORY = "UL Group/Image Generation"
    FUNCTION = "sample"
    TITLE = "AnyText Sampler"
    DESCRIPTION = "AnyText: Multilingual Visual Text Generation And Editing.\nAnyText comprises a diffusion pipeline with two primary elements: an auxiliary latent module and a text embedding module. The former uses inputs like text glyph, position, and masked image to generate latent features for text generation or editing. The latter employs an OCR model for encoding stroke data as embeddings, which blend with image caption embeddings from the tokenizer to generate texts that seamlessly integrate with the background. We employed text-control diffusion loss and text perceptual loss for training to further enhance writing accuracy.\nAnyText多语言文字生成与编辑\n通过创新性的算法设计，可以支持中文、英语、日语、韩语等多语言的文字生成，还支持对输入图片中的文字内容进行编辑。本模型所涉及的文字生成技术为电商海报、Logo设计、创意涂鸦、表情包等新型AIGC应用提供了可能性。\nAnyText主要基于扩散（Diffusion）模型，包含两个核心模块：隐空间辅助模块（Auxiliary Latent Module）和文本嵌入模块（Text Embedding Module）。其中，隐空间辅助模块对三类辅助信息（字形、文字位置和掩码图像）进行编码并构建隐空间特征图像，用来辅助视觉文字的生成；文本嵌入模块则将描述词中的语义部分与待生成文本的字形部分解耦，使用图像编码模块单独提取字形信息后再与语义信息做融合，既有助于文字的书写精度，也有利于提升文字与背景的一致性。训练阶段，除了使用扩散模型常用的噪声预测损失，我们还增加了文本感知损失，在图像空间对每个生成文本区域进行像素级的监督，以进一步提升文字书写精度。"

    def sample(self, model, positive, negative, seed, steps, cfg, strength, eta, keep_model_loaded, keep_model_device):
        model['model'].model.diffusion_model.to(get_torch_device())
        model['model'].control_model.to(get_torch_device())
        
        from pytorch_lightning import seed_everything
        from .AnyText_scripts.cldm.ddim_hacked import DDIMSampler
        seed_everything(seed)
        
        model['model'].control_scales = ([strength] * 13)
        
        ddim_sampler = DDIMSampler(model['model'], device=get_torch_device())
        pbar = ProgressBar(steps)
        def callback(*_):
            pbar.update(1)
        latents, intermediates = ddim_sampler.sample(
            S=steps, 
            batch_size=positive[0][1]['batch_size'],
            shape=positive[0][1]['shape'], 
            conditioning=positive[0][0], 
            verbose=False, 
            eta=eta,
            unconditional_guidance_scale=cfg,
            unconditional_conditioning=negative[0][0],
            callback=callback, #后端代码有timesteps的callback
        )
        
        if not keep_model_loaded:
            del model['model'].model.diffusion_model
            del model['model'].control_model
            del model['model'].cond_stage_model
            del model['model'].embedding_manager
        elif keep_model_device and keep_model_loaded and torch.cuda.is_available():
            model['model'].model.diffusion_model.to(unet_offload_device())
            model['model'].control_model.to(text_encoder_offload_device())
            model['model'].embedding_manager.to(text_encoder_offload_device())
        
        # model['model'].first_stage_model.to(get_torch_device())
        
        # samples = model['model'].decode_first_stage(latents)
        # samples = (einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        # if not keep_model_loaded:
        #     del model['model'].first_stage_model
        #     del model['model']
        #     soft_empty_cache(True)
        #     clean_up()
        # elif keep_model_loaded and torch.cuda.is_available():
        #     model['model'].first_stage_model.to(vae_offload_device())
        #     soft_empty_cache(True)
        #     clean_up()
        
        # result = []
        # for sample in samples:
        #     result.append(pil2tensor(sample))
        # result = torch.cat(result, dim=0)
        
        # return(result, {"samples": 1. / 0.18215 * latents})
        return({"samples": 1. / 0.18215 * latents}, )

class UL_AnyTextLoader:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "control_net_name": (["None"] + folder_paths.get_filename_list("controlnet"), ),
                "miaobi_clip": (["None"] + folder_paths.get_filename_list("text_encoders"), ),
                "weight_dtype": (["auto", "fp16", "fp32", "bf16", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto", "tooltip": "Only fp16 and fp32 works.\n仅支持fp16和fp32。"}),
                }
            }

    RETURN_TYPES = ("AnyText_Model", "VAE", "STRING", )
    RETURN_NAMES = ("model", "vae", "ckpt_name", )
    FUNCTION = "Loader"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Loader"
    DESCRIPTION = "Miaobi_clip is optional, for chinese prompt text_encode without translator.\nOption 1: load full AnyText ckeckpoint in ckpt_name without controlnet.\nOption 2: load custom sd1.5 ckeckpoint with AnyText control_net.\nmiaobi_clip是可选项，用于输入中文提示词但不使用翻译机。\n选项1： 加载完整的AnyText模型，此时勿加载control_net。\n选项2：加载自定义sd1.5模型和AnyText的control_net。"

    def Loader(self, ckpt_name, control_net_name, miaobi_clip, weight_dtype):
        from .AnyText_scripts.cldm.model import create_model
    
        dtype = get_dtype_by_name(weight_dtype)
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_yaml', 'anytext_sd15.yaml')
        
        model = create_model(cfg_path, use_fp16=(dtype == torch.float16)) #dtype control
        state_dict = load_torch_file(ckpt_path, safe_load=True)
        
        if control_net_name != "None":
            anytext_controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
            anytext_state_dict = load_torch_file(anytext_controlnet_path, safe_load=True)
            for k in list(anytext_state_dict):
                state_dict[k] = anytext_state_dict[k]
                anytext_state_dict[k] = None
            # state_dict.update(anytext_state_dict)
            del anytext_state_dict
          
        if miaobi_clip != "None":
            from transformers import AutoTokenizer
            custom_tokenizer =  AutoTokenizer.from_pretrained(MiaoBi_tokenizer_dir, trust_remote_code=True)
            model.cond_stage_model.tokenizer = custom_tokenizer
            clip_path = folder_paths.get_full_path_or_raise("text_encoders", miaobi_clip)
            for k in list(state_dict.keys()):
                if k.startswith("cond_stage_model"):
                    state_dict[k] = None
                    state_dict.pop(k)
            clip_l_sd = load_torch_file(clip_path, safe_load=True)
        else:  
            clip_l_sd = {}
            for k in list(state_dict.keys()):
                if k.startswith("cond_stage_model"):
                    clip_l_sd[k.replace("cond_stage_model.transformer.", "")] = state_dict[k]
                    state_dict.pop(k)
        
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        model.cond_stage_model.transformer.load_state_dict(clip_l_sd, strict=False)
        del clip_l_sd
        model.cond_stage_model.freeze()
        clean_up()
        
        model.eval().to(get_torch_device(), dtype)
        
        model = {
            'model': model,
        }
        
        return (model, VAE(copy.deepcopy(model["model"].first_stage_model)), os.path.basename(ckpt_path), )

class UL_AnyTextInputs:
    @classmethod
    def INPUT_TYPES(self):
        self.font_files = os.listdir(os.path.join(folder_paths.models_dir, "fonts"))
        return {
            "required": {
                "font_name": (['Auto_DownLoad'] + self.font_files, {"default": "AnyText-Arial-Unicode.ttf"}),
                "apply_translate": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "translator": (["utrobinmv/t5_translate_en_ru_zh_small_1024", "damo/nlp_csanmt_translation_zh2en", "utrobinmv/t5_translate_en_ru_zh_base_200", "utrobinmv/t5_translate_en_ru_zh_large_1024"],{"default": "utrobinmv/t5_translate_en_ru_zh_small_1024", "tooltip": "Translate models for zh2en.\n中译英模型，t5_small体积小(212MB)但质量一般，nlp体积大(7.35GB)质量高但是需要自行安装依赖`ComfyUI\custom_nodes\ComfyUI_Anytext\requirements-with-nlp-translator.txt`，其余不建议。"}), 
                "translator_device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                "keep_translator_loaded": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "keep_translator_device": ("BOOLEAN", {"default": True, "label_on": "comfy", "label_off": "device"}),
                "Auto_Download_Path": ("BOOLEAN", {"default": True, "label_on": "models_local本地", "label_off": ".cache缓存", "tooltip": "Cache translator model files to huggingface cache_dir or download into `ComfyUI\models\prompt_generator`.\n缓存翻译模型到huggingface缓存路径或者下载到`ComfyUI\models\prompt_generator`。"}),
                }
            }

    RETURN_TYPES = ("AnyText_Inputs", "STRING", )
    RETURN_NAMES = ("inputs", "font_name", )
    FUNCTION = "inputs"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Inputs"
    DESCRIPTION = ""

    def inputs(self, font_name, apply_translate, translator, translator_device, keep_translator_loaded, keep_translator_device, Auto_Download_Path, show_debug=False):
        font_path = os.path.join(folder_paths.models_dir, "fonts", font_name)
        from PIL import ImageFont
        if font_name == "Auto_DownLoad":
            font_path = os.path.join(folder_paths.models_dir, "fonts", "SourceHanSansSC-Medium.otf")
            if not os.path.exists(font_path):
                from huggingface_hub import hf_hub_download as hg_hf_hub_download
                hg_hf_hub_download(
                    repo_id="Sanster/AnyText", 
                    filename="SourceHanSansSC-Medium.otf", 
                    local_dir=os.path.join(folder_paths.models_dir, "fonts"), 
                    )
                
        font = ImageFont.truetype(font_path, size=60, encoding='utf-8')
        
        translator_device = get_device_by_name(translator_device)
        
        inputs = {
            "font": font,
            "apply_translate": apply_translate,
            "translator": translator,
            "show_debug": show_debug,
            "translator_device": translator_device,
            "keep_translator_loaded": keep_translator_loaded,
            "keep_translator_device": keep_translator_device,
            "Auto_Download_Path": Auto_Download_Path,
        }
        
        return (inputs, os.path.basename(font_path), )


class UL_AnyTextFontImg:
    @classmethod
    def INPUT_TYPES(self):
        self.font_files = os.listdir(os.path.join(folder_paths.models_dir, "fonts"))
        return {
            "required": {
                "font": (['Auto_DownLoad'] + self.font_files, {"default": "索尼兰亭.ttf"}),
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

    def FontImg(self, font, pos_mask, prompt, width, height, sort_radio, font_color_name, font_color_code, font_color_mode, bg_color_name, bg_color_code, bg_color_mode, font_color_codeR, font_color_codeG, font_color_codeB, font_color_codeA, seperate_by):
        texts = str(prompt).split(seperate_by)
        n_lines = len(texts)
        if len(texts ) == 0:
            texts  = [' ']
        max_chars = 50
        font_path = os.path.join(folder_paths.models_dir, "fonts", font)
        self.font = ImageFont.truetype(font_path, size=60, encoding='utf-8')
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
                _, glyphs = draw_glyph2(self.font, text, poly_list[i], scale=gly_scale, width=width, height=height, add_space=False, font_color=font_color)
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
        return {
            "required": {
                "model": ("AnyText_Model", ),
                "inputs": ("AnyText_Inputs", ),
                "mask": ("MASK", ),
                "prompt": ("STRING", {"forceInput": True}),
                "texts": ("LIST", ),
                "latent": ("LATENT", ),
                "mode": ("BOOLEAN", {"default": True, "label_on": "text-generation生成", "label_off": "text-editing文字编辑"}),
                "sort_radio": ("BOOLEAN", {"default": True, "label_on": "↔水平", "label_off": "↕垂直", "tooltip": "Order of draw texts according to mask position orders. ↕ for y axis. It will draw text-content(“string”) from start-to-end(order) on the mask position from top to bottom. ↔ for x axis .It will draw text-content(“string”) from start-to-end(order) on the mask position from left to right.\n根据遮罩位置顺序决定生成文本的顺序。"}),
                "a_prompt": ("STRING", {"default": "best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks", "multiline": True}),
                "n_prompt": ("STRING", {"default": "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture", "multiline": True}),
                "revise_pos": ("BOOLEAN", {"default": False, "tooltip": "Which uses the bounding box of the rendered text as the revised position. However, it is occasionally found that the creativity of the generated text is slightly lower using this method, It dosen’t work in text-edit mode.\n使用边界盒子渲染文字作位置调整。但是发现偶尔会影响生成质量，仅在使用随机生成遮罩时生效。"}),
                "random_mask": ("BOOLEAN", {"default": False, "tooltip": "Random generate mask, the input mask will be ignored.\n随机生成遮罩，输入的遮罩将被忽略。"}),
            },
            "optional": {
                "image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE", "LATENT", )
    RETURN_NAMES = ("positive", "negative", "mask_img", "masked_x", )
    FUNCTION = "encoder"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Encoder"
    
    def encoder(self, model, inputs, mask, prompt, texts, latent, mode, sort_radio, a_prompt, n_prompt, revise_pos, random_mask, image=None):
        model['model'].control_model.to(text_encoder_offload_device())
        model['model'].model.diffusion_model.to(text_encoder_offload_device())
        
        from .AnyText_scripts.AnyText_pipeline import separate_pos_imgs, find_polygon, draw_glyph, draw_glyph2
        from .AnyText_scripts.AnyText_pipeline_util import check_channels
        
        max_chars = 50
        batch_size, height, width = latent["samples"].shape[0], latent["samples"].shape[2] * 8, latent["samples"].shape[3] * 8 # B, C, H, W
        
        dtype = model['model'].dtype
        
        font = inputs["font"]
        translator = inputs["translator"]
        Auto_Download_Path = inputs["Auto_Download_Path"]
        
        #check if prompt is chinese to decide whether to load translator，检测是否为中文提示词，否则不适用翻译。
        is_chinese = check_chinese(prompt_replace(prompt))
        if inputs['apply_translate'] and is_chinese:
            #如果启用中译英，则提前判断本地是否存在翻译模型，没有则自动下载，以防跑半路报错。
            # elif loader_out[3] == 'utrobinmv/t5_translate_en_ru_zh_small_1024':
            if translator == 'utrobinmv/t5_translate_en_ru_zh_small_1024':
                base_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_small_1024")
                if Auto_Download_Path and not os.path.exists(os.path.join(base_path, "model.safetensors")):
                    download_repoid_model_from_huggingface(repo_id=translator, Base_Path=base_path)
                    
            elif translator == 'damo/nlp_csanmt_translation_zh2en':
                from modelscope.hub.snapshot_download import snapshot_download as modelscope_snapshot_download
                base_path = os.path.join(folder_paths.models_dir, "prompt_generator", "modelscope--damo--nlp_csanmt_translation_zh2en")
                if Auto_Download_Path and not os.path.exists(os.path.join(base_path, "tf_ckpts", "ckpt-0.data-00000-of-00001")):
                    modelscope_snapshot_download(translator, local_dir=base_path)
                    
            # elif loader_out[3] == 'utrobinmv/t5_translate_en_ru_zh_base_200':
            elif translator == 'utrobinmv/t5_translate_en_ru_zh_base_200':
                base_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_base_200")
                if Auto_Download_Path and not os.path.exists(os.path.join(base_path, "model.safetensors")):
                    download_repoid_model_from_huggingface(translator, base_path)
                    
            # elif loader_out[3] == 'utrobinmv/t5_translate_en_ru_zh_large_1024':
            elif translator == 'utrobinmv/t5_translate_en_ru_zh_large_1024':
                base_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_large_1024")
                if Auto_Download_Path and not os.path.exists(os.path.join(base_path, "model.safetensors")):
                    download_repoid_model_from_huggingface(translator, base_path)
                    
        # tensor图片转换为numpy图片
        pos_image = tensor2numpy_cv2(mask)
        pos_image = resize_image(pos_image, max(width, height))
        pos_image = cv2.cvtColor(pos_image, cv2.COLOR_GRAY2RGB) # cv2二值图(mask)转rgb
        pos_image = cv2.bitwise_not(pos_image) # cv2图片取反
        
        if mode:
            mode = 'text-generation'
            revise_pos = revise_pos
        else:
            if image == None:
                raise ValueError('Edit mode need a image.')
            ori_image = tensor2numpy_cv2(image)
            ori_image = resize_image(ori_image, max(width, height))
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
            mode = 'text-editing'
            revise_pos = False
        
        n_lines = len(texts)
        h, w = height, width
        
        if random_mask == True:
            mask = generate_rectangles(width, height, n_lines, max_trys=500)
            pos_image = cv2.imread(Random_Gen_Mask_path)
            if n_lines == 1 or n_lines % 2 ==0:
                mask = pos_image
            mask = numpy_cv2tensor(mask)
        else:
            pos_image = pos_image
            mask = numpy_cv2tensor(pos_image)
        
        sort_radio = '↔' if sort_radio else '↕'
        
        anytext_prompt = prompt
        
        if anytext_prompt is None and texts is None:
            raise ValueError("Invalid prompt: 无效提示词。")
        
        if mode in ['text-generation', 'gen']:
            if random_mask:
                edit_image = np.ones((h, w, 3)) * 127.5  # empty mask image
                edit_image = resize_image(edit_image, max_length=max(h, w))
                h, w = edit_image.shape[:2]
            else:
                edit_image = pos_image[..., ::-1]
                edit_image = resize_image(edit_image, max_length=max(h, w))
                h, w = edit_image.shape[:2]
                edit_image = np.ones((h, w, 3)) * 127.5  # empty mask image
        elif mode in ['text-editing', 'edit']:
            if pos_image is None or ori_image is None:
                return None, -1, "Reference image and position image are needed for text editing!", ""
            if isinstance(ori_image, np.ndarray):
                ori_image = ori_image[..., ::-1]
                assert ori_image is not None, f"Can't read ori_image image from{ori_image}!"
            elif isinstance(ori_image, torch.Tensor):
                ori_image = ori_image.cpu().numpy()
                
            edit_image = ori_image.clip(1, 255)  # for mask reason
            edit_image = check_channels(edit_image)
            edit_image = resize_image(edit_image, max_length=max(h, w))  # make w h multiple of 64, resize if w or h > max_length
            h, w = edit_image.shape[:2]  # change h, w by input ref_img
        # preprocess pos_imgs(if numpy, make sure it's white pos in black bg)
        if pos_image is None:
            pos_imgs = np.zeros((w, h, 1))
        if isinstance(pos_image, np.ndarray):
            pos_image = pos_image[..., ::-1]
            pos_image = resize_image(pos_image, max_length=max(h, w))
            pos_image = cv2.resize(pos_image, (w, h))
            assert pos_image is not None, f"Can't read pos_image image from{pos_image}!"
            pos_imgs = 255-pos_image
        elif isinstance(pos_image, torch.Tensor):
            pos_imgs = pos_image.cpu().numpy()
            
        pos_imgs = pos_imgs[..., 0:1]
        pos_imgs = cv2.convertScaleAbs(pos_imgs)
        _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
        # seprate pos_imgs
        pos_imgs = separate_pos_imgs(pos_imgs, sort_radio)
        if len(pos_imgs) == 0:
            pos_imgs = [np.zeros((h, w, 1))]
        if len(pos_imgs) < n_lines:
            if n_lines == 1 and texts[0] == ' ':
                # pass  # text-to-image without text
                print('\033[93m', f'Warning: text-to-image without text.', '\033[0m')
            else:
                raise ValueError(f'Found {len(pos_imgs)} positions that < needed {n_lines} from prompt, check and try again(手绘遮罩数少于要绘制的文本数，检查修改再重试)!')
        elif len(pos_imgs) > n_lines:
            print('\033[93m', f'Warning: found {len(pos_imgs)} positions that > needed {n_lines} from prompt.', '\033[0m')
            
        # get pre_pos, poly_list, hint that needed for anytext
        pre_pos = []
        poly_list = []
        for input_pos in pos_imgs:
            if input_pos.mean() != 0:
                input_pos = input_pos[..., np.newaxis] if len(input_pos.shape) == 2 else input_pos
                poly, pos_img = find_polygon(input_pos)
                pre_pos += [pos_img/255.]
                poly_list += [poly]
            else:
                pre_pos += [np.zeros((h, w, 1))]
                poly_list += [None]
        np_hint = np.sum(pre_pos, axis=0).clip(0, 1)
        
        # prepare info dict
        info = {}
        info['glyphs'] = []
        info['gly_line'] = []
        info['positions'] = []
        info['n_lines'] = [len(texts)]*batch_size
        gly_pos_imgs = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > max_chars:
                print(f'"{text}" length > max_chars: {max_chars}, will be cut off...')
                text = text[:max_chars]
            gly_scale = 2
            if pre_pos[i].mean() != 0:
                gly_line = draw_glyph(font, text)
                glyphs, _ = draw_glyph2(font, text, poly_list[i], scale=gly_scale, width=w, height=h, add_space=False)
                gly_pos_img = cv2.drawContours(glyphs*255, [poly_list[i]*gly_scale], 0, (255, 255, 255), 1)
                if revise_pos:
                    resize_gly = cv2.resize(glyphs, (pre_pos[i].shape[1], pre_pos[i].shape[0]))
                    new_pos = cv2.morphologyEx((resize_gly*255).astype(np.uint8), cv2.MORPH_CLOSE, kernel=np.ones((resize_gly.shape[0]//10, resize_gly.shape[1]//10), dtype=np.uint8), iterations=1)
                    new_pos = new_pos[..., np.newaxis] if len(new_pos.shape) == 2 else new_pos
                    contours, _ = cv2.findContours(new_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if len(contours) != 1:
                        print(f'Fail to revise position {i} to bounding rect, remain position unchanged...')
                    else:
                        rect = cv2.minAreaRect(contours[0])
                        poly = np.int0(cv2.boxPoints(rect))
                        pre_pos[i] = cv2.drawContours(new_pos, [poly], -1, 255, -1) / 255.
                        gly_pos_img = cv2.drawContours(glyphs*255, [poly*gly_scale], 0, (255, 255, 255), 1)
                gly_pos_imgs += [gly_pos_img]  # for show
            else:
                glyphs = np.zeros((h*gly_scale, w*gly_scale, 1))
                gly_line = np.zeros((80, 512, 1))
                gly_pos_imgs += [np.zeros((h*gly_scale, w*gly_scale, 1))]  # for show
            pos = pre_pos[i]
            info['glyphs'] += [arr2tensor(glyphs, batch_size, dtype)]
            info['gly_line'] += [arr2tensor(gly_line, batch_size, dtype)]
            info['positions'] += [arr2tensor(pos, batch_size, dtype)]
            
        # get masked_x
        masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0)*(1-np_hint)
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().to(get_torch_device(), dtype)
        
        model['model'].first_stage_model.to(get_torch_device())
        encoder_posterior = model['model'].encode_first_stage(masked_img[None, ...])
        masked_x = (model['model'].get_first_stage_encoding(encoder_posterior).detach()).to(dtype)
        model['model'].first_stage_model.to(text_encoder_offload_device())
        
        info['masked_x'] = torch.cat([masked_x for _ in range(batch_size)], dim=0)
        hint = arr2tensor(np_hint, batch_size, dtype)
        
        model['model'].cond_stage_model.to(get_torch_device())
        model['model'].embedding_manager.to(get_torch_device())
        cond = model["model"].get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[anytext_prompt + ' , ' + a_prompt] * batch_size], text_info=info))
        un_cond = model["model"].get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[n_prompt] * batch_size], text_info=info))
        model['model'].cond_stage_model.to(text_encoder_offload_device())
        model['model'].embedding_manager.to(text_encoder_offload_device())
        
        if torch.cuda.is_available():
            soft_empty_cache(True)
            clean_up()
        
        return ([[cond, {"pooled_output": {}, "shape": (4, h // 8, w // 8), "batch_size": batch_size}]], [[un_cond, {"pooled_output": {}}]], mask, {"samples": 1. / 0.18215 * masked_x})
    
class UL_AnyTextFormatter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputs": ("AnyText_Inputs", ),
                "prompt": ("STRING", {"default": 'close-up of hakurei reimu sitting in a room, with text: "博丽灵梦" on the wall.', "multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                
            }
        }

    RETURN_TYPES = ("STRING", "LIST", )
    RETURN_NAMES = ("prompt", "texts", )
    FUNCTION = "formatter"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Formatter"
    
    def formatter(self, inputs, prompt):
        prompt, texts = modify_prompt(
            prompt=prompt,
            translator=inputs['translator'],
            Auto_Download_Path=inputs['Auto_Download_Path'],
            translator_device=inputs['translator_device'],
            keep_translator_loaded=inputs['keep_translator_loaded'],
            keep_translator_device=inputs['keep_translator_device'],
            apply_translate=inputs['apply_translate'],
        )
        return (prompt, texts)

class UL_AnyTextLoaderTest:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "control_net_name": (["None"] + folder_paths.get_filename_list("controlnet"), ),
                "miaobi_clip": (["None"] + folder_paths.get_filename_list("text_encoders"), ),
                "weight_dtype": (["auto", "fp16", "fp32", "bf16", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto", "tooltip": "Only fp16 and fp32 works.\n仅支持fp16和fp32。"}),
                },
            "optional": {
                "c_model" :("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("AnyText_Model", "VAE", "STRING", )
    RETURN_NAMES = ("model", "vae", "ckpt_name", )
    FUNCTION = "Loader"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Loader(Test)"
    DESCRIPTION = "Only LoRA without trigger word works.\nIf not input c_model, then load ckpt_name model.\nMiaobi_clip is optional, for chinese prompt text_encode without translator.\nOption 1: load full AnyText ckeckpoint in ckpt_name without controlnet.\nOption 2: load custom sd1.5 ckeckpoint with AnyText control_net.\n仅没有触发词的LoRA能生效。\n如果不输入c_model，则从ckpt_name加载模型。\nmiaobi_clip是可选项，用于输入中文提示词但不使用翻译机。\n选项1： 加载完整的AnyText模型，此时勿加载control_net。\n选项2：加载自定义sd1.5模型和AnyText的control_net。"
    
    def __init__(self):
        self.raised = False

    def Loader(self, ckpt_name, control_net_name, miaobi_clip, weight_dtype, c_model=None, clip=None, vae=None):
        if c_model != None and control_net_name == "None" and not self.raised:
            self.raised = True
            raise ValueError("If input c_model, AnyText Control model is essential, or text mask will not work.\n如果输入c_model，则必须输入AnyText Control model，否则文字遮罩无效。")
            
        from .AnyText_scripts.cldm.model import create_model
    
        dtype = get_dtype_by_name(weight_dtype)
        
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_yaml', 'anytext_sd15.yaml')
        
        model = create_model(cfg_path, use_fp16=(dtype == torch.float16)) #dtype control
        
        if c_model == None:
            state_dict = load_torch_file(ckpt_path, safe_load=True)
        else:
            load_model_gpu(c_model)
            state_dict = c_model.model.state_dict_for_saving(None, vae.get_sd(), None)
            soft_empty_cache(True)
        
        if control_net_name != "None":
            anytext_controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
            anytext_state_dict = load_torch_file(anytext_controlnet_path, safe_load=True)
            for k in list(anytext_state_dict):
                state_dict[k] = anytext_state_dict[k]
                anytext_state_dict[k] = None
            # state_dict.update(anytext_state_dict)
            del anytext_state_dict
          
        if miaobi_clip != "None":
            from transformers import AutoTokenizer
            custom_tokenizer =  AutoTokenizer.from_pretrained(MiaoBi_tokenizer_dir, trust_remote_code=True)
            model.cond_stage_model.tokenizer = custom_tokenizer
            clip_path = folder_paths.get_full_path_or_raise("text_encoders", miaobi_clip)
            for k in list(state_dict.keys()):
                if k.startswith("cond_stage_model"):
                    state_dict[k] = None
                    state_dict.pop(k)
            clip_l_sd = load_torch_file(clip_path, safe_load=True)
        elif clip != None and miaobi_clip == "None":
            clip_model = clip.load_model()
            load_model_gpu(clip_model)
            clip_l_sd = clip.get_sd()
            clip_l_sd = c_model.model.model_config.process_clip_state_dict_for_saving(clip_l_sd)
            soft_empty_cache(True)
            for k in list(clip_l_sd.keys()):
                if k.startswith("cond_stage_model"):
                    clip_l_sd[k.replace("cond_stage_model.transformer.", "")] = clip_l_sd[k]
                    clip_l_sd[k] = None
                    clip_l_sd.pop(k)
        elif c_model == None and miaobi_clip == "None":
            clip_l_sd = {}
            for k in list(state_dict.keys()):
                if k.startswith("cond_stage_model"):
                    clip_l_sd[k.replace("cond_stage_model.transformer.", "")] = state_dict[k]
                    state_dict.pop(k)
        
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        model.cond_stage_model.transformer.load_state_dict(clip_l_sd, strict=False)
        del clip_l_sd
        model.cond_stage_model.freeze()
        if c_model != None:
            comfy_clean_vram()
        clean_up()
        
        model.eval().to(get_torch_device(), dtype)
        
        model = {
            'model': model,
        }
        
        return (model, VAE(copy.deepcopy(model["model"].first_stage_model.to(vae_offload_device()))) if vae==None else vae, os.path.basename(ckpt_path), )

# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "UL_AnyText_Sampler": UL_AnyTextSampler,
    "UL_AnyText_Inputs": UL_AnyTextInputs,
    "UL_AnyText_Loader": UL_AnyTextLoader,
    "UL_AnyText_FontImg": UL_AnyTextFontImg,
    "UL_AnyText_Composer": UL_AnyTextComposer,
    "UL_AnyTextEncoder": UL_AnyTextEncoder,
    "UL_AnyTextFormatter": UL_AnyTextFormatter,
    "UL_AnyTextLoaderTest": UL_AnyTextLoaderTest,
}


def check_overlap_polygon(rect_pts1, rect_pts2):
            poly1 = cv2.convexHull(rect_pts1)
            poly2 = cv2.convexHull(rect_pts2)
            rect1 = cv2.boundingRect(poly1)
            rect2 = cv2.boundingRect(poly2)
            if rect1[0] + rect1[2] >= rect2[0] and rect2[0] + rect2[2] >= rect1[0] and rect1[1] + rect1[3] >= rect2[1] and rect2[1] + rect2[3] >= rect1[1]:
                return True
            return False
        
def count_lines(prompt):
    prompt = prompt.replace('“', '"').replace('”', '"')
    p = '"(.*?)"'
    strs = re.findall(p, prompt)
    if len(strs) == 0:
        strs = [' ']
    return len(strs)

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

def generate_rectangles(w, h, n, max_trys=200):
    img = np.zeros((h, w, 1), dtype=np.uint8)
    rectangles = []
    attempts = 0
    n_pass = 0
    low_edge = int(max(w, h)*0.3 if n <= 3 else max(w, h)*0.2)  # ~150, ~100
    while attempts < max_trys:
        rect_w = min(np.random.randint(max((w*0.5)//n, low_edge), w), int(w*0.8))
        ratio = np.random.uniform(4, 10)
        rect_h = max(low_edge, int(rect_w/ratio))
        rect_h = min(rect_h, int(h*0.8))
        # gen rotate angle
        rotation_angle = 0
        rand_value = np.random.rand()
        if rand_value < 0.7:
            pass
        elif rand_value < 0.8:
            rotation_angle = np.random.randint(0, 40)
        elif rand_value < 0.9:
            rotation_angle = np.random.randint(140, 180)
        else:
            rotation_angle = np.random.randint(85, 95)
        # rand position
        x = np.random.randint(0, w - rect_w)
        y = np.random.randint(0, h - rect_h)
        # get vertex
        rect_pts = cv2.boxPoints(((rect_w/2, rect_h/2), (rect_w, rect_h), rotation_angle))
        rect_pts = np.int32(rect_pts)
        # move
        rect_pts += (x, y)
        # check boarder
        if np.any(rect_pts < 0) or np.any(rect_pts[:, 0] >= w) or np.any(rect_pts[:, 1] >= h):
            attempts += 1
            continue
        # check overlap
        if any(check_overlap_polygon(rect_pts, rp) for rp in rectangles): # type: ignore
            attempts += 1
            continue
        n_pass += 1
        img = cv2.fillPoly(img, [rect_pts], 255)
        cv2.imwrite(Random_Gen_Mask_path, 255-img[..., ::-1])
        rectangles.append(rect_pts)
        if n_pass == n:
            break
        if n >2 and n % 2 != 0:
            img += img
            img = cv2.imread(Random_Gen_Mask_path)
        print("attempts:", attempts)
    if len(rectangles) != n:
        raise Exception(f'Failed in auto generate positions after {attempts} attempts, try again!')
    return img

# def replace_between(s, start, end, replacement):
#     # 正则表达式，用以匹配从start到end之间的所有字符
#     pattern = r"%s(.*?)%s" % (re.escape(start), re.escape(end))
#     # 使用re.DOTALL标志来匹配包括换行在内的所有字符
#     return re.sub(pattern, replacement, s, flags=re.DOTALL)

def check_chinese(text):
    from .AnyText_scripts.AnyText_bert_tokenizer import BasicTokenizer
    checker = BasicTokenizer()
    text = checker._clean_text(text)
    for char in text:
        cp = ord(char)
        if checker._is_chinese_char(cp):
            return True
    return False

def modify_prompt(prompt, translator, Auto_Download_Path, translator_device, keep_translator_loaded, keep_translator_device, apply_translate):
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
    if check_chinese(prompt) and apply_translate:
        if translator is None:
            return None, None
        old_prompt = prompt
        
        from ..Data_Process.utils import t5_translate_en_ru_zh, nlp_csanmt_translation_zh2en
        
        if translator == 'utrobinmv/t5_translate_en_ru_zh_small_1024':
            zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_small_1024")
            if not os.path.exists(os.path.join(zh2en_path, "model.safetensors")) and not Auto_Download_Path:
                zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_small_1024'
            prompt, _, _ = t5_translate_en_ru_zh('en', prompt + ' .', zh2en_path, translator_device, keep_translator_loaded, keep_model_device=keep_translator_device)
            prompt = prompt[0]
            
        elif translator == 'utrobinmv/t5_translate_en_ru_zh_base_200':
            zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_base_200")
            if not os.path.exists(os.path.join(zh2en_path, "model.safetensors")) and not Auto_Download_Path:
                zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_base_200'
            prompt, _, _ = t5_translate_en_ru_zh('en', prompt + ' .', zh2en_path, translator_device, keep_translator_loaded, keep_model_device=keep_translator_device)
            prompt = prompt[0]
            
        elif translator == 'utrobinmv/t5_translate_en_ru_zh_large_1024':
            zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_large_1024")
            if not os.path.exists(os.path.join(zh2en_path, "model.safetensors")) and not Auto_Download_Path:
                zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
            prompt, _, _ = t5_translate_en_ru_zh('en', prompt + ' .', zh2en_path, translator_device, keep_translator_loaded, keep_model_device=keep_translator_device)
            prompt = prompt[0]
            
        else:
            nlp_device = 'gpu'
            if 'cpu' in translator_device.type:
                nlp_device = 'cpu'
            zh2en_path = os.path.join(folder_paths.models_dir, 'prompt_generator', 'modelscope--damo--nlp_csanmt_translation_zh2en')
            if not os.path.exists(os.path.join(zh2en_path, "tf_ckpts", "ckpt-0.data-00000-of-00001")) and not Auto_Download_Path:
                zh2en_path = "damo/nlp_csanmt_translation_zh2en"
            prompt = nlp_csanmt_translation_zh2en(nlp_device, prompt + ' .', zh2en_path)['translation']
        print(f'Translate: {old_prompt} --> {prompt}')
    return prompt, strs

def arr2tensor(arr, bs, dtype):
        arr = np.transpose(arr, (2, 0, 1))
        _arr = torch.from_numpy(arr.copy()).float().to(get_torch_device())
        _arr = (torch.stack([_arr for _ in range(bs)], dim=0)).to(dtype)
        return _arr
    
class VAE:
    def __init__(self, vae):
        self.vae = vae
    
    def decode(self, latent):
        self.vae.to(get_torch_device())
        samples = self.vae.decode(latent)
        self.vae.to(vae_offload_device())
        if torch.cuda.is_available():
            soft_empty_cache(True)
            clean_up()
        
        samples = (einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        result = []
        for sample in samples:
            result.append(pil2tensor(sample))
        result = torch.cat(result, dim=0)
        
        return result