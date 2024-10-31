import os
import folder_paths
import re
import cv2
import numpy as np
import torch
from PIL import ImageFont, Image
from comfy.utils import load_torch_file
from .AnyText_scripts.AnyText_pipeline_util import resize_image
from ..UL_common.common import pil2tensor, get_device_by_name, get_files_with_extension, tensor2numpy_cv2, download_repoid_model_from_huggingface, tensor2pil, numpy_cv2tensor, Pillow_Color_Names, clean_up, get_dtype_by_name
from .. import comfy_temp_dir

Random_Gen_Mask_path = os.path.join(comfy_temp_dir,  "AnyText_random_mask_pos_img.png")

class UL_Image_Generation_AnyText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anytext_model": ("AnyText_Model", ),
                "anytext_params": ("AnyText_Params", ),
                "prompt": ("STRING", {"default": "A realistic photo with text “UL Group” in the background, a richly decorated cake with cream “阿里云” and “APSARA”。", "multiline": True}),
                "a_prompt": ("STRING", {"default": "best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks", "multiline": True}),
                "n_prompt": ("STRING", {"default": "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture", "multiline": True}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "text-generation文字生成", "label_off": "text-editing文字编辑"}),  
                "sort_radio": ("BOOLEAN", {"default": True, "label_on": "↔水平", "label_off": "↕垂直", "tooltip": "Order of draw texts according to mask position orders. ↕ for y axis. It will draw text-content(“string”) from start-to-end(order) on the mask position from top to bottom. ↔ for x axis .It will draw text-content(“string”) from start-to-end(order) on the mask position from left to right.\n根据遮罩位置顺序决定生成文本的顺序。"}), 
                "ddim_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 88888888, "min": 0, "max": 4294967296}),
                "seed": ("INT", {"default": 88888888, "min": -1, "max": np.iinfo(np.int32).max}),
                "width": ("INT", {"default": 512, "min": 128, "max": 3840, "step": 64}),
                "height": ("INT", {"default": 512, "min": 128, "max": 3840, "step": 64}),
                "revise_pos": ("BOOLEAN", {"default": False, "tooltip": "Which uses the bounding box of the rendered text as the revised position. However, it is occasionally found that the creativity of the generated text is slightly lower using this method, It dosen’t work in text-edit mode.\n使用边界盒子渲染文字作位置调整。但是发现偶尔会影响生成质量，仅在使用随机生成遮罩时生效。"}),
                "Random_Gen": ("BOOLEAN", {"default": False, "tooltip": "Random generate mask, the input mask will be ignored.\n随机生成遮罩，输入的遮罩将被忽略。"}),
                "strength": ("FLOAT", {"default": 1.00, "min": 0, "max": 2, "step": 0.01}),
                "cfg_scale": ("FLOAT", { "default": 9, "min": 1, "max": 99, "step": 0.1}),
                "eta": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto", "tooltip": "Pytorch devices."}), 
                "keep_model_loaded": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Warning: do not delete model unless this node no longer needed, it will try release device_memory and ram. if checked and want to continue node generation, use ComfyUI-Manager `Free model and node cache` to reset node state or change parameter in Loader node to activate.\n注意：仅在这个节点不再需要时删除模型，将尽量释放系统内存和设备专用内存。如果删除后想继续使用此节点，使用ComfyUI-Manager插件的`Free model and node cache`重置节点状态或者更换模型加载节点的参数来激活。"}),
                "keep_model_device": ("BOOLEAN", {"default": True, "label_on": "cpu", "label_off": "device", "tooltip": "Keep model in ram or device_memory after generation.\n生图完成后，模型转移到系统内存或者保留在设备专用内存上。"}),
            },
            "optional": {
                        "ori_image": ("IMAGE", {"forceInput": True}),
                        "pos_mask": ("MASK", {"forceInput": True}),
                        },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", )
    RETURN_NAMES = ("image", "mask", )
    CATEGORY = "UL Group/Image Generation"
    FUNCTION = "UL_Image_Generation_AnyText"
    TITLE = "AnyText Sampler"
    DESCRIPTION = "AnyText: Multilingual Visual Text Generation And Editing.\nAnyText comprises a diffusion pipeline with two primary elements: an auxiliary latent module and a text embedding module. The former uses inputs like text glyph, position, and masked image to generate latent features for text generation or editing. The latter employs an OCR model for encoding stroke data as embeddings, which blend with image caption embeddings from the tokenizer to generate texts that seamlessly integrate with the background. We employed text-control diffusion loss and text perceptual loss for training to further enhance writing accuracy.\nAnyText多语言文字生成与编辑\n通过创新性的算法设计，可以支持中文、英语、日语、韩语等多语言的文字生成，还支持对输入图片中的文字内容进行编辑。本模型所涉及的文字生成技术为电商海报、Logo设计、创意涂鸦、表情包等新型AIGC应用提供了可能性。\nAnyText主要基于扩散（Diffusion）模型，包含两个核心模块：隐空间辅助模块（Auxiliary Latent Module）和文本嵌入模块（Text Embedding Module）。其中，隐空间辅助模块对三类辅助信息（字形、文字位置和掩码图像）进行编码并构建隐空间特征图像，用来辅助视觉文字的生成；文本嵌入模块则将描述词中的语义部分与待生成文本的字形部分解耦，使用图像编码模块单独提取字形信息后再与语义信息做融合，既有助于文字的书写精度，也有利于提升文字与背景的一致性。训练阶段，除了使用扩散模型常用的噪声预测损失，我们还增加了文本感知损失，在图像空间对每个生成文本区域进行像素级的监督，以进一步提升文字书写精度。"
  
    def __init__(self):
        self.loaded_prompt = None
        self.processed_prompt = None
        self.texts = None
        self.model = None

    def UL_Image_Generation_AnyText(
        self,
        anytext_model,
        anytext_params,
        mode,
        ori_image,
        pos_mask,
        sort_radio,
        revise_pos,
        Random_Gen,
        prompt, 
        batch_size, 
        device,
        keep_model_loaded,
        keep_model_device,
        ddim_steps=20, 
        strength=1, 
        cfg_scale=9, 
        seed="", 
        eta=0.0, 
        a_prompt="", 
        n_prompt="", 
        width=512, 
        height=512,
    ):    
        device = get_device_by_name(device)  
        dtype = anytext_model['model'].dtype
        
        self.model = anytext_model['model']
        if 'cpu' in anytext_model['model'].device.type:
            self.model.to(device)
        
        font = anytext_params["font"]
        translator = anytext_params["translator"]
        show_debug = anytext_params["show_debug"]
        Auto_Download_Path = anytext_params["Auto_Download_Path"]
        
        #调需要用时才进行导包，减少启动时的加载时间。
        from .AnyText_scripts.AnyText_pipeline import AnyText_Pipeline
        
        #check if prompt is chinese to decide whether to load translator，检测是否为中文提示词，否则不适用翻译。
        prompt_modify = prompt_replace(prompt)
        bool_is_chinese = AnyText_Pipeline.is_chinese(self, prompt_modify)
        if bool_is_chinese == True:
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
                
        pipe = AnyText_Pipeline(
            anytext_model=self.model, 
            loaded_prompt=self.loaded_prompt,
            processed_prompt=self.processed_prompt,
            texts=self.texts,
            translator=translator, 
            use_translator=bool_is_chinese, 
            model_device=device, 
            model_dtype=dtype,
            translator_device=anytext_params["translator_device"], 
            keep_translator_loaded=anytext_params["keep_translator_loaded"], 
            keep_translator_device=anytext_params["keep_translator_device"],
            Auto_Download_Path=anytext_params['Auto_Download_Path'],
            )
        
        # tensor图片转换为numpy图片
        pos_image = tensor2numpy_cv2(pos_mask)
        pos_image = resize_image(pos_image, max(width, height))
        pos_image = cv2.cvtColor(pos_image, cv2.COLOR_GRAY2RGB) # cv2二值图(mask)转rgb
        pos_image = cv2.bitwise_not(pos_image) # cv2图片取反
        
        ori_image = tensor2numpy_cv2(ori_image)
        ori_image = resize_image(ori_image, max(width, height))
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
            
        if mode:
            mode = 'text-generation'
            ori_image = None
            revise_pos = revise_pos
        else:
            mode = 'text-editing'
            revise_pos = False
            
        n_lines = count_lines(prompt)
        if Random_Gen == True:
            mask = generate_rectangles(width, height, n_lines, max_trys=500)
            pos_img = cv2.imread(Random_Gen_Mask_path)
            if n_lines == 1 or n_lines % 2 ==0:
                mask = pos_img
            # pos_img = cv2.cvtColor(pos_img, cv2.COLOR_GRAY2RGB)
            # pos_img = cv2.bitwise_not(pos_img)
            mask = numpy_cv2tensor(mask)
        else:
            pos_img = pos_image
            mask = numpy_cv2tensor(pos_image)
            
        # lora_path = r"D:\AI\ComfyUI_windows_portable\ComfyUI\models\loras\ys艺术\sd15_mw_bpch_扁平风格插画v1d1.safetensors"
        # lora_ratio = 1
        # lora_path_ratio = str(lora_path)+ " " + str(lora_ratio)
        # print("\033[93m", lora_path_ratio, "\033[0m")
        
        if ddim_steps == 10 or ddim_steps == 1:
            ddim_steps = ddim_steps + 1
        
        if sort_radio:
            sort_radio = '↔'
        else:
            sort_radio = '↕'
        
        params = {
            "mode": mode,
            "translator_device": anytext_params["translator_device"],
            "Random_Gen": Random_Gen,
            "sort_priority": sort_radio,
            "revise_pos": revise_pos,
            "show_debug": show_debug,
            "image_count": batch_size,
            "ddim_steps": ddim_steps - 1,
            "image_width": width,
            "image_height": height,
            "strength": strength,
            "cfg_scale": cfg_scale,
            "eta": eta,
            "a_prompt": a_prompt,
            "n_prompt": n_prompt,
            # "lora_path_ratio": lora_path_ratio,
            }
        
        input_data = {
                "prompt": prompt,
                "seed": seed,
                "draw_pos": pos_img,
                "ori_image": ori_image,
                }
        
        if show_debug ==True:
            print(f'\033[93mloader from .util(从.util输入的anytext_params): {params}, \033[0m\n \
                    \033[93mIs Chinese, True will load translator(是否中文输入,是的话加载翻译): {bool_is_chinese}, \033[0m\n \
                    \033[93mTranslator(翻译模型)--loader_out[3]: {translator}, \033[0m\n \
                    \033[93mNumber of text-content to generate(需要生成的文本数量): {n_lines}, \033[0m\n \
                    \033[93mSort Position(文本生成位置排序): {sort_radio}, \033[0m\n \
                    \033[93mEnable revise_pos(启用位置修正): {revise_pos}, \033[0m')
        x_samples, results, rtn_code, rtn_warning, debug_info, self.processed_prompt, self.texts, self.loaded_prompt = pipe(
            font=font, 
            input_tensor=input_data, 
            **params)
        
        if rtn_code < 0:
            raise Exception(f"Error in AnyText pipeline: {rtn_warning}")
        
        result = []
        for samples in x_samples:
            result.append(pil2tensor(samples))
        result = torch.cat(result, dim=0)
            
        print("\n", debug_info)
        
        if not keep_model_loaded:
            del anytext_model['model']
            self.model = None
            del pipe.anytext_model
            del pipe
            clean_up()
        else:
            if keep_model_device:
                self.model.to('cpu')
                clean_up()
        
        return(result, mask, )

class AnyText_Model_Loader:
    @classmethod
    def INPUT_TYPES(self):
        checkpoints_list = folder_paths.get_filename_list("checkpoints")
        clip_list = os.listdir(os.path.join(folder_paths.models_dir, "clip"))
        clip_folders = [folder for folder in clip_list if os.path.isdir(os.path.join(folder_paths.models_dir, "clip", folder))]
        return {
            "required": {
                "ckpt_name": (['Auto_DownLoad'] + checkpoints_list, {"tooltip": "Must be AnyText pretained checkpoint, other SD1.5 base model doesn't work. If Auto_Download selected, fp16 checkpoint will download from huggingface into `ComfyUI\models\checkpoints\15` and rename to `anytext_v1.1.safetensors`.\n只支持AnyText预训练模型，其他SD1.5模型无效。如果选择自动下载(Auto_DownLoad)且以前没下载过，基座模型会下载到`ComfyUI\models\checkpoints\15`。"}),
                "clip": (["Auto_DownLoad"] + clip_folders, {"tooltip": "If Auto_Download selected, clip model files will cached (Auto_Download_Path not checked) or download into `ComfyUI\models\clip` (Auto_Download_Path checked).\n如果选择自动下载(Auto_DownLoad)且以前没下载过并且勾选(Auto_Download_Path)下载到本地，clip模型文件将下载到`ComfyUI\models\clip`，否则缓存到huggingface缓存路径。"}),
                "dtype": (["auto", "fp16", "fp32", "bf16", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto", "tooltip": "Now only fp16 and fp32 works.\n现在仅支持fp16和fp32。"}),
                "Auto_Download_Path": ("BOOLEAN", {"default": True, "label_on": "models_local本地", "label_off": ".cache缓存", "tooltip": "Cache clip model files to huggingface cache_dir or download into `ComfyUI\models\clip`.\nclip模型自动下载位置选择：huggingface缓存路径或者`ComfyUI\models\clip`。"}),
                # "unet_for_merge": ("STRING", {"default": r"D:\AI\ComfyUI_windows_portable\ComfyUI\models\diffusers\models--SG161222--Realistic_Vision_V6.0_B1_noVAE\unet\diffusion_pytorch_model.bin", "multiline": False, "dynamicPrompts": False}), 
                # "save_merged_dir": ("STRING", {"default": r"C:\Users\pc\Desktop\merged_AnyText_unet.safetensors", "multiline": False, "dynamicPrompts": False}), 
                # "merge_unet": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                }
            }

    RETURN_TYPES = ("AnyText_Model", "STRING", )
    RETURN_NAMES = ("anytext_model", "ckpt_name", )
    FUNCTION = "Loader"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Model Loader"

    # def UL_Image_Generation_AnyText_Params(self, ckpt_name, clip, translator, dtype, save_merged_dir, merge_model, unet_for_merge):
    def Loader(self, ckpt_name, clip, dtype, Auto_Download_Path):
        dtype = get_dtype_by_name(dtype)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_yaml', 'anytext_sd15.yaml')
        clip_path = os.path.join(folder_paths.models_dir, "clip", clip)
        if not os.path.exists(clip_path):
            if Auto_Download_Path:
                clip_path = os.path.join(folder_paths.models_dir, "clip", 'models--openai--clip-vit-large-patch14')
                if not os.path.exists(os.path.join(clip_path, 'model.safetensors')):
                    download_repoid_model_from_huggingface("openai/clip-vit-large-patch14", clip_path, ignore_patterns=[".msgpack", ".bin", ".h5"])
            else:
                clip_path = "openai/clip-vit-large-patch14"
                
        if ckpt_path == None:
            ckpt_path = os.path.join(folder_paths.models_dir, "checkpoints", "15", "anytext_v1.1.safetensors")
            if not os.path.exists(ckpt_path):
                from huggingface_hub import hf_hub_download as hg_hf_hub_download
                hg_hf_hub_download(
                    repo_id="Sanster/AnyText", 
                    filename="pytorch_model.fp16.safetensors",
                    local_dir = os.path.join(folder_paths.models_dir, "checkpoints", "15"),
                    )
                
                old_file = os.path.join(folder_paths.models_dir, "checkpoints", "15", "pytorch_model.fp16.safetensors")
                os.rename(old_file, ckpt_path)
        
        from .AnyText_scripts.cldm.model import create_model
        
        model = create_model(cfg_path, cond_stage_path=clip_path, use_fp16=(dtype == torch.float16))
        state_dict = load_torch_file(ckpt_path, safe_load=True)
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        model.eval().to(dtype)
        
        # if merge_model:
        #     from .AnyText_tools.tool_add_anytext import create_anytext_model
        #     create_anytext_model(unet_for_merge, save_merged_dir, cfg_path, 'cpu)
        
        # return (params, save_merged_dir, )
        
        model = {
            'model': model,
        }
        
        return (model, os.path.basename(ckpt_path), )

class AnyText_Params:
    @classmethod
    def INPUT_TYPES(self):
        self.font_files = get_files_with_extension('fonts', ['.ttf', '.otf'])
        return {
            "required": {
                "font_name": (['Auto_DownLoad'] + [file for file in self.font_files], {"default": "AnyText-Arial-Unicode.ttf"}),
                "translator": (["utrobinmv/t5_translate_en_ru_zh_small_1024", "damo/nlp_csanmt_translation_zh2en", "utrobinmv/t5_translate_en_ru_zh_base_200", "utrobinmv/t5_translate_en_ru_zh_large_1024"],{"default": "utrobinmv/t5_translate_en_ru_zh_small_1024", "tooltip": "Translate models for zh2en.\n中译英模型，t5_small体积小(212MB)但质量一般，nlp体积大(7.35GB)质量高但是需要自行安装tensorflow和modelscope库，其余不建议。"}), 
                "translator_device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                "keep_translator_loaded": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "keep_translator_device": ("BOOLEAN", {"default": True, "label_on": "cpu", "label_off": "device"}),
                "Auto_Download_Path": ("BOOLEAN", {"default": True, "label_on": "models_local本地", "label_off": ".cache缓存", "tooltip": "Cache translator model files to huggingface cache_dir or download into `ComfyUI\models\prompt_generator`.\n缓存翻译模型到huggingface缓存路径或者下载到`ComfyUI\models\prompt_generator`。"}),
                }
            }

    RETURN_TYPES = ("AnyText_Params", "STRING", )
    RETURN_NAMES = ("anytext_params", "font_name", )
    FUNCTION = "AnyText_Params"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Params"
    DESCRIPTION = ""

    def AnyText_Params(self, font_name, translator, translator_device, keep_translator_loaded, keep_translator_device, Auto_Download_Path, show_debug=False):
        font_path = os.path.join(folder_paths.models_dir, "fonts", font_name)
        from PIL import ImageFont
        if not os.path.exists(font_path):
            font_path = os.path.join(folder_paths.models_dir, "fonts", "AnyText-SourceHanSansSC-Medium.otf")
            if not os.path.exists(font_path):
                from huggingface_hub import hf_hub_download as hg_hf_hub_download
                hg_hf_hub_download(
                    repo_id="Sanster/AnyText", 
                    filename="SourceHanSansSC-Medium.otf", 
                    local_dir=os.path.join(folder_paths.models_dir, "fonts"), force_filename='AnyText-SourceHanSansSC-Medium.otf',
                    )
                
        font = ImageFont.truetype(font_path, size=60, encoding='utf-8')
        
        translator_device = get_device_by_name(translator_device)
        
        params = {
            "font": font,
            "translator": translator,
            "show_debug": show_debug,
            "translator_device": translator_device,
            "keep_translator_loaded": keep_translator_loaded,
            "keep_translator_device": keep_translator_device,
            "Auto_Download_Path": Auto_Download_Path,
        }
        
        if show_debug == True:
            print(f'\033[93mloader(合并后的输入参数字典，传递给nodes): {params} \033[0m\n \
                    \033[93mfont_path(字体): {font_path} \033[0m\n \
                    \033[93mtranslator_path(翻译模型): {translator} \033[0m\n'
                    )
        return (params, os.path.basename(font_path), )


class UL_Image_Generation_AnyText_Font_Img:
    @classmethod
    def INPUT_TYPES(self):
        self.font_files = get_files_with_extension('fonts', ['.ttf', '.otf'])
        return {
            "required": {
                "font": (['Auto_DownLoad'] + [file for file in self.font_files], {"default": "索尼兰亭.ttf"}),
                "pos_mask": ("MASK", ),
                # "sort_radio": (["↕", "↔"], {"default": "↔"}), 
                "sort_radio": ("BOOLEAN", {"default": True, "label_on": "↔水平排序", "label_off": "↕垂直排序"}), 
                "font_color_name": (['transparent'] + Pillow_Color_Names, {"default": "white"}),
                # "font_color_name": (Pillow_Color_Names, {"default": "white"}),
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
    FUNCTION = "UL_Image_Generation_AnyText_Font_Img"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Font_Img"

    def UL_Image_Generation_AnyText_Font_Img(self, font, pos_mask, prompt, width, height, sort_radio, font_color_name, font_color_code, font_color_mode, bg_color_name, bg_color_code, bg_color_mode, font_color_codeR, font_color_codeG, font_color_codeB, font_color_codeA, seperate_by):
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
        draw_pos = resize_image(mask_img, max_length=768)
        draw_pos = cv2.resize(draw_pos, (width, height))
        pos_imgs = 255-draw_pos
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
                
        # print('\033[93m', f'poly_list: {poly_list}', '\n', f'type: {type(pos_imgs)}', '\033[0m')
        
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
        
class UL_Image_Generation_AnyText_Font_Img_Composer:
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
    FUNCTION = "UL_Image_Generation_AnyText_Font_Img_Composer"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "AnyText Composer"
    
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,)

    def UL_Image_Generation_AnyText_Font_Img_Composer(self, mode, font_or_bg_img=None, font_img2=None, font_img3=None, font_img4=None, font_img5=None, font_img6=None, font_img7=None, font_img8=None, font_img9=None, font_img10=None):
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
        
# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "UL_Image_Generation_AnyText": UL_Image_Generation_AnyText,
    "UL_Image_Generation_AnyText_Params": AnyText_Params,
    "UL_Image_Generation_AnyText_Model_Loader": AnyText_Model_Loader,
    "UL_Image_Generation_AnyText_Font_Img": UL_Image_Generation_AnyText_Font_Img,
    "UL_Image_Generation_AnyText_Font_Img__Composer": UL_Image_Generation_AnyText_Font_Img_Composer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UL_Image_Generation_AnyText": "Img_Gen AnyText",
    "UL_Image_Generation_AnyText_Params": "Img_Gen AnyText Params",
    "UL_Image_Generation_AnyText_Font_Img": "Img_Gen AnyText Font_Img",
    "UL_Image_Generation_AnyText_Font_Img_Compose": "Img_Gen AnyText Composer",
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