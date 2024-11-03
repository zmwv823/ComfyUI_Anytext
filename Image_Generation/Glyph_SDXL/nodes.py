import os
import torch
import json
import numpy as np
from copy import deepcopy
import webcolors
from comfy.utils import load_torch_file
from comfy.model_management import unet_offload_device
from ...UL_common.common import get_device_by_name, clean_up, tensor2numpy_cv2, Scheduler_Names, SDXL_Scheduler_List, cv2pil, seperate_masks, tensor2pil, download_repoid_model_from_huggingface
import folder_paths
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
MAX_TEXT_BOX = 20
    
class UL_Image_Generation_Glyph_SDXL_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "glyph_sdxl_model": ("Glyph_SDXL_Model", ),
                "font_params": ("Glyph_SDXL_Font_Params", ),
                "bg_prompt": ("STRING", {"default": "背景图提示词\nThe image features a green and blue globe with a factory on top of it. The factory is surrounded by trees, giving the impression of a harmonious coexistence between the industrial structure and the natural environment. The globe is prominently displayed in the center of the image, with the factory and trees surrounding it.\nor\nThe image features a beautiful young 20yo woman sitting in a field of flowers, holding a little cute girl in her arms. The scene is quite serene and picturesque, with the two human being the main focus of the image. The field is filled with various flowers, creating a beautiful and vibrant backdrop for the humans.", "multiline": True, "dynamicPrompts": True, "tooltip": "背景图提示词。"}),
                # "bg_class": ("STRING", {"default": "背景类型(可选)\nPosters\nor\nCards and invitations", "multiline": True, "dynamicPrompts": True, "tooltip": "背景类型。"}), 
                "bg_class": ("STRING", {"default": "背景类型---Posters or Cards and invitations", "tooltip": "背景类型。"}), 
                "bg_tags": ("STRING", {"default": "背景描述(可选)\ngreen, modern, earth, world, planet, ecology, background, globe, environment, day, space, map, concept, global, light, hour, energy, power, protect, illustration\nor\nLight green, orange, Illustration, watercolor, playful, Baby shower invitation, baby boy shower invitation, baby boy, welcoming baby boy, koala baby shower invitation, baby shower invitation for baby shower, baby boy invitation, background, playful baby shower card, baby shower, card, newborn, born, Baby Shirt Baby Shower Invitation", "multiline": True, "dynamicPrompts": True, "tooltip": "背景描述。"}), 
                "negative_prompt": ("STRING", {"default": "worst quality, watermark, author name.", "multiline": True, "dynamicPrompts": True, "tooltip": "负面词。"}),
                "lora_scale": ("FLOAT", {"default": 1, "min": -9.00, "max": 9.00, "step": 0.01, "tooltip": "Text strength."}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "generation", "label_off": "inpaint", "tooltip": "Inpaint not work well, official edit code not released."}),
                "mask": ("MASK", ),
                "mask_gap": ("INT", {"default": 50,"min": 0, "max": 10240, "step": 1, "tooltip": "Seperate masks from input mask."}),
                "sort_radio": ("BOOLEAN", {"default": True, "label_on": "↔水平", "label_off": "↕垂直", "tooltip": "控制生成文字的位置顺序，根据遮罩的顺序确认文字对应位置的顺序。水平则从左往右，最靠近画布左边的遮罩位置先开始，垂直则从上往下，最靠近画布上边的遮罩位置开始。"}), 
                # "width": ("INT", {"default": 1024,"min": 512, "max": 10240, "step": 1}),
                # "height": ("INT", {"default": 1024,"min": 512, "max": 10240, "step": 1}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 99, "step": 1, "tooltip": "Standard scheduler needs more than 20 steps, for tcd scheduler set to 8 steps."}),
                "cfg": ("FLOAT", {"default": 1.5, "min": 0.00, "max": 10, "step": 0.01, "tooltip": "Default 5 for standard scheduler, for TCD with lightning model: 1~1.5."}),
                "scheduler": (Scheduler_Names, {"default": "TCD", "tooltip": "Schedulers."}),
                # "seed": ("INT", {"default": 0,"min": 0, "max": 2147483647, "step": 1, "tooltip": "Seed for control."}),
                "seed": ("INT", {"default": 88888888,"min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "tooltip": "Seed for control."}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto"}), 
                "keep_model_loaded": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Warning: do not delete model unless this node no longer needed, it will try release device_memory and ram. if checked and want to continue node generation, use ComfyUI-Manager `Free model and node cache` to reset node state or change parameter in Loader node to activate.\n注意：仅在这个节点不再需要时删除模型，将尽量释放系统内存和设备专用内存。如果删除后想继续使用此节点，使用ComfyUI-Manager插件的`Free model and node cache`重置节点状态或者更换模型加载节点的参数来激活。"}),
                "keep_model_device": ("BOOLEAN", {"default": True, "label_on": "comfy", "label_off": "device", "tooltip": "Keep model in comfy_auto_unet_offload_device (HIGH_VRAM: device, Others: cpu) or device_memory after generation.\n生图完成后，模型转移到comfy自动选择设备(HIGH_VRAM: device, 其他: cpu)或者保留在设备专用内存上。"}),
                },
            "optional":{
                "vae": ("VAE", {"tooltip": "VAE input for inpaint."}),
                "inpaint_image": ("IMAGE", {"tooltip": "Image for ip-adapter in generation mode or for inpaint in inpaint mode."}),
                "inp_strength": ("FLOAT", {"default": 0.55,"min": 0, "max": 1.00, "step": 0.01, "tooltip": "Inpaint strength."}),
                "masked_latent": ("LATENT", {"tooltip": "Masked image latent for inpaint, if input then mask will be ignored."}),
                "clip_vision": (['None', 'Vit-H', 'Vit-bigG', 'Dino v2'], {"default": 'None', "tooltip": "Image encoder (ip-adapter) for inpaint, Vit-H: plus, `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` and Vit-bigG: `CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` in `ComfyUI\models\clip_vision`. Seems not work"}),
                }
            }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "UL_Image_Generation_Glyph_SDXL"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "Glyph-SDXL Sampler"
    DESCRIPTION = "Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering\nWe identify two crucial requirements of text encoders for achieving accurate visual text rendering: character awareness and alignment with glyphs. To this end, we propose a customized text encoder, Glyph-ByT5, by fine-tuning the character-aware ByT5 encoder using a meticulously curated paired glyph-text dataset.\nWe present an effective method for integrating Glyph-ByT5 with SDXL, resulting in the creation of the Glyph-SDXL model for design image generation. This significantly enhances text rendering accuracy, improving it from less than 20% to nearly 90% on our design image benchmark. Noteworthy is Glyph-SDXL's newfound ability for text paragraph rendering, achieving high spelling accuracy for tens to hundreds of characters with automated multi-line layouts.\nWe deliver a powerful customized multilingual text encoder, Glyph-ByT5-v2, and a strong aesthetic graphic generation model, Glyph-SDXL-v2, that can support accurate spelling in ∼10 different languages"
    
    def __init__(self) -> None:
        self.mode = None
        self.vae = None
        self.byt5_text_encoder = None
        self.origin_module = None
        self.Glyph_SDXL_pipe = None
        self.loaded_lora_scale = None
        self.loaded_prompt_and_bboxes = None
        self.loaded_embeds = None
        self.loaded_attn_masks_dicts = None
        self.loaded_ip_adapter_img = None
        self.loaded_image_embeds = None
        self.loaded_clip_vision = None
        
    def UL_Image_Generation_Glyph_SDXL(self, seed, cfg, scheduler, mask, font_params, glyph_sdxl_model, bg_prompt, lora_scale, device, keep_model_loaded, keep_model_device, steps=50,  bg_class='', bg_tags='', sort_radio='↔', width=1024, height=1024, mask_gap=102, masked_latent=None, inpaint_image=None, mode=True, inp_strength=0.999, clip_vision='Vit-H', negative_prompt=None, vae=None, debug=False,debug_only=False):
        
        device = get_device_by_name(device)
        
        if not mode and glyph_sdxl_model['diffusers_model']['unet'].config.in_channels != 9:
            print('Use non-inpaint checkpoint for inpainting.')
        
        color_ann_path = os.path.join(current_dir, 'Glyph_SDXL_Scripts', 'assets', 'color_idx.json')
        font_ann_path = os.path.join(current_dir, 'Glyph_SDXL_Scripts', 'assets', 'font_idx_512.json')
        if glyph_sdxl_model['version']:
            font_ann_path = os.path.join(current_dir, 'Glyph_SDXL_Scripts', 'assets', 'multilingual_10-lang_idx.json')
        from .Glyph_SDXL_Scripts.glyph_sdxl.utils.format_prompt import MultilingualPromptFormat
        multilingual_prompt_format = MultilingualPromptFormat(font_ann_path, color_ann_path)
        
        mask = tensor2numpy_cv2(mask)
        mask = cv2.resize(mask, dsize=(width, height))
        
        sort_radio = '↔' if sort_radio else '↕'
        seperated_masks = seperate_masks(mask, sort_radio, gap=mask_gap)
        
        if len(seperated_masks) < 1:
            raise ValueError(f'Need at least one mask.')
        
        multilingual_stack = []
        for i, masks in enumerate(seperated_masks):
            y_coords, x_coords = np.nonzero(masks)
            #     # x_min = x_coords.min()
            #     # x_max = x_coords.max()
            #     # y_min = y_coords.min()
            #     # y_max = y_coords.max()
            #     # mask_width = x_max - x_min # 遮罩宽度
            #     # mask_height = y_max - y_min # 遮罩高度
            stack = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]
            multilingual_stack.append(stack)
        
        if sort_radio == '↕': # 根据列表中第几个值大小，由小到大重新排序
            def takeSecond(elem):
                return elem[1] # 第二个值y_min
            multilingual_stack.sort(key=takeSecond)
        if sort_radio == '↔':
            def takeSecond(elem):
                return elem[0] # 第一个值，x_min
            multilingual_stack.sort(key=takeSecond)
        
        conditions = font_params
        stack_cp = deepcopy(multilingual_stack)
        
        prompts = []
        colors = []
        font_type = []
        langs = []
        bboxes = []
        num_boxes = len(stack_cp) if len(stack_cp[-1]) == 4 else len(stack_cp) - 1
        for i in range(num_boxes):
            if conditions[i + MAX_TEXT_BOX * 2] is None:
                raise ValueError(f"Invalid conditions for box {i + 1} !")
            
        for i in range(num_boxes):
            prompts.append(conditions[i])
            colors.append(conditions[i + MAX_TEXT_BOX])
            lang = conditions[i + MAX_TEXT_BOX * 2].split(":")[0].strip()
            font = conditions[i + MAX_TEXT_BOX * 2].split(":")[1].strip()
            print(conditions[i + MAX_TEXT_BOX * 2], " ", lang, " ", font)
            langs.append(multilingual_reverse_code_dict[lang])
            font_type.append(f'{multilingual_reverse_code_dict[lang]}-{font}')
            
        styles = []
        if bg_prompt == "" or bg_prompt is None:
            raise ValueError("Empty background prompt!")
        
        for i, (font_text, color, style) in enumerate(zip(prompts, colors, font_type)):
            if font_text == "" or font_text is None:
                raise ValueError(f"Invalid prompt for text box {i + 1} !")
            if color is None:
                raise ValueError(f"Invalid color for text box {i + 1} !")
            if style is None:
                raise ValueError(f"Invalid style for text box {i + 1} !")
            
            bboxes.append(
                [
                    stack_cp[i][0] / 1024,
                    stack_cp[i][1] / 1024,
                    (stack_cp[i][2] - stack_cp[i][0]) / 1024,
                    (stack_cp[i][3] - stack_cp[i][1]) / 1024,
                ]
            )
            
            styles.append(
                {
                    'color': webcolors.name_to_hex(color),
                    'font-family': style,
                }
            )
        
        # 3. format input
        if bg_class != "":# and bg_class is not None:
            bg_prompt = bg_class + ". " + bg_prompt
        if bg_tags != "":# and bg_tags is not None:
            bg_prompt += " Tags: " + bg_tags
        
        final_font_text = multilingual_prompt_format.format_prompt(prompts, styles)
        
        if debug and debug_only:
            print('\033[93m', '-----Debug info before pipe input-----', '\033[0m')
            print('\033[93m', f'stack_cp :{stack_cp}', '\n', f'type: {type(stack_cp)}', '\033[0m')
            print('\033[93m', f'conditions :{conditions}', '\n', f'type: {type(conditions)}', '\033[0m')
            print('\033[93m', f'multilingual_stack :{multilingual_stack}', '\n', f'type: {type(multilingual_stack)}', '\033[0m')
            print('\033[93m', f'bg_tags :{bg_tags}', '\n', f'type: {type(bg_tags)}', '\033[0m')
            print('\033[93m', f'bg_class :{bg_class}', '\n', f'type: {type(bg_class)}', '\033[0m')
            print('\033[93m', f"styles: {styles}", '\033[0m')
            print('\033[93m', f"prompt: {bg_prompt}", '\033[0m')
            print('\033[93m', f"text_prompt: {final_font_text}", '\033[0m')
            print('\033[93m', f"texts: {prompts}", '\033[0m')
            print('\033[93m', f'bboxes :{bboxes}', '\n', f'type: {type(bboxes)}', '\033[0m')
        
        
        if not debug_only:
            if debug:
                if mode:
                    print('\033[93m', 'Start generation.', '\033[0m')
                else:
                    print('\033[93m', 'Start inpaint.', '\033[0m')
            from .Glyph_SDXL_Scripts.glyph_sdxl.custom_diffusers.pipelines.pipeline_stable_diffusion_glyph_xl import StableDiffusionGlyphXLPipeline
            
            pipe = StableDiffusionGlyphXLPipeline
            self.vae = glyph_sdxl_model['diffusers_model']['vae']
            if not mode:
                from .Glyph_SDXL_Scripts.glyph_sdxl.custom_diffusers.pipelines.pipeline_stable_diffusion_glyph_xl_inpaint import StableDiffusionGlyphXLPipeline_Inpaint
                
                pipe = StableDiffusionGlyphXLPipeline_Inpaint
                
                from diffusers import AutoencoderKL
                from ...UL_common.sdxl_singlefile_model_config import SDXL_VAE_fp16_fix_config
                from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
                
                state_dict = vae.get_sd()
                state_dict = convert_ldm_vae_checkpoint(state_dict, SDXL_VAE_fp16_fix_config)
                vae = AutoencoderKL(**SDXL_VAE_fp16_fix_config)
                vae.load_state_dict(state_dict, strict=False)
                del state_dict
                self.vae.to(glyph_sdxl_model['diffusers_model']['unet'].dtype)
            
            self.Glyph_SDXL_pipe = pipe(
                vae=self.vae,
                tokenizer=glyph_sdxl_model['diffusers_model']['clip']['tokenizer'],
                tokenizer_2=glyph_sdxl_model['diffusers_model']['clip']['tokenizer_2'],
                text_encoder=glyph_sdxl_model['diffusers_model']['clip']['text_encoder'],
                text_encoder_2=glyph_sdxl_model['diffusers_model']['clip']['text_encoder_2'],
                byt5_text_encoder=glyph_sdxl_model['byt5_text_encoder'],
                byt5_tokenizer=glyph_sdxl_model['byt5_tokenizer'],
                byt5_mapper=glyph_sdxl_model['byt5_mapper'],
                byt5_max_length=glyph_sdxl_model['byt5_max_length'],
                unet=glyph_sdxl_model['diffusers_model']['unet'],
                scheduler = SDXL_Scheduler_List()[scheduler],
            )
        
            # with torch.cuda.amp.autocast():
            generator = torch.Generator(device=device).manual_seed(int(seed))
            
            negative_prompt = None if negative_prompt == '' else negative_prompt
            
            if debug:
                print('\033[93m', '-----Debug info before pipe input-----', '\033[0m')
                print('\033[93m', f'stack_cp :{stack_cp}', '\n', f'type: {type(stack_cp)}', '\033[0m')
                print('\033[93m', f'conditions :{conditions}', '\n', f'type: {type(conditions)}', '\033[0m')
                print('\033[93m', f'multilingual_stack :{multilingual_stack}', '\n', f'type: {type(multilingual_stack)}', '\033[0m')
                print('\033[93m', f'bg_tags :{bg_tags}', '\n', f'type: {type(bg_tags)}', '\033[0m')
                print('\033[93m', f'bg_class :{bg_class}', '\n', f'type: {type(bg_class)}', '\033[0m')
                print('\033[93m', f"styles: {styles}", '\033[0m')
                print('\033[93m', f"prompt: {bg_prompt}", '\033[0m')
                print('\033[93m', f"text_prompt: {final_font_text}", '\033[0m')
                print('\033[93m', f"texts: {prompts}", '\033[0m')
                print('\033[93m', f'bboxes :{bboxes}', '\n', f'type: {type(bboxes)}', '\033[0m')
                
            with torch.amp.autocast(device.type):
                if not mode:
                    inpaint_image = tensor2pil(inpaint_image)
                    inpaint_image = inpaint_image.resize(size=[width, height])
                    cords = stack_cp[0]
                    ipadapter_img = inpaint_image # multi masks
                    if len(seperated_masks) == 1: # single mask
                        if debug:
                            print('\033[93m', '\nSingle mask inpaint.\n', '\033[0m')
                        ipadapter_img = inpaint_image.crop(box=[cords[0], cords[1], cords[2], cords[3]])
                    mask = cv2pil(mask)
                    if debug:
                        from ... import system_desktop_dir
                        inpaint_image.save(os.path.join(system_desktop_dir, 'inpaint_img.png'))
                        ipadapter_img.save(os.path.join(system_desktop_dir, 'ipadapter_img.png'))
                        mask.save(os.path.join(system_desktop_dir, 'mask_img.png'))
                    
                    masked_image_latents = masked_latent['samples'] if masked_latent != None else None
                else:
                    ipadapter_img = inpaint_image
                    mask = None
                    masked_image_latents = None
                    inp_strength = None
                
                image, self.loaded_lora_scale, self.loaded_prompt_and_bboxes, self.loaded_embeds, self.loaded_attn_masks_dicts, self.loaded_ip_adapter_img, self.loaded_image_embeds, self.loaded_clip_vision = self.Glyph_SDXL_pipe(
                    width=width,
                    height=height,
                    debug=debug,
                    prompt=bg_prompt,
                    text_prompt=final_font_text,
                    texts=prompts,
                    bboxes=bboxes,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
                    text_attn_mask=None,
                    device=device,
                    cross_attention_kwargs={"scale": lora_scale},
                    negative_prompt=negative_prompt,
                    # inpaint params
                    image=inpaint_image,
                    ip_adapter_image=ipadapter_img,
                    mask_image=mask,
                    masked_image_latents=masked_image_latents,
                    strength=inp_strength,
                    image_encoder_type=clip_vision,
                    output_type="latent", # 如果加这个参数，输出为原始latent，在comfyui输出需要除以vae.config.scaling_factor,然后latent = {"samples": latent},不加这个参数则直接输出PIL image.
                    # 返回生成的数据防止重复处理
                    loaded_lora_scale=self.loaded_lora_scale,
                    loaded_prompt_and_bboxes=self.loaded_prompt_and_bboxes,
                    loaded_embeds=self.loaded_embeds,
                    loaded_attn_masks_dicts=self.loaded_attn_masks_dicts,
                    loaded_ip_adapter_img=self.loaded_ip_adapter_img,
                    loaded_image_embeds=self.loaded_image_embeds,
                    loaded_clip_vision=self.loaded_clip_vision
                )#.images[0] # 输出PIL图片需要的参数
            
            if not keep_model_loaded:
                del glyph_sdxl_model['diffusers_model']['unet']
                del glyph_sdxl_model['diffusers_model']['vae']
                del glyph_sdxl_model['diffusers_model']['clip']
                del glyph_sdxl_model['diffusers_model']['scheduler']
                del glyph_sdxl_model['diffusers_model']['pipe']
                del glyph_sdxl_model['diffusers_model']
                del glyph_sdxl_model['byt5_text_encoder']
                del glyph_sdxl_model['byt5_mapper']
                del glyph_sdxl_model['byt5_tokenizer']
                del self.Glyph_SDXL_pipe.unet
                del self.Glyph_SDXL_pipe.vae
                del self.Glyph_SDXL_pipe.tokenizer
                del self.Glyph_SDXL_pipe.tokenizer_2
                del self.Glyph_SDXL_pipe.text_encoder
                del self.Glyph_SDXL_pipe.text_encoder_2
                del self.Glyph_SDXL_pipe.byt5_mapper
                del self.Glyph_SDXL_pipe.byt5_text_encoder
                del self.Glyph_SDXL_pipe.byt5_tokenizer
                del self.Glyph_SDXL_pipe.scheduler
                del self.Glyph_SDXL_pipe
                del self.vae
                del self.byt5_text_encoder
                clean_up()
            else:
                if keep_model_device:
                    glyph_sdxl_model['diffusers_model']['unet'].to(unet_offload_device())
                    clean_up()
        else:
            # import random
            # from PIL import Image
            # color_list = random.sample(range(0,255),4)
            # image = Image.new(mode='RGBA', size=[768, 768], color=(color_list[0], color_list[1], color_list[2], color_list[3]))
            
            latent = torch.zeros([1, 4, height // 8, width // 8], device=device) # 创建empty latent供调试。
            image = {"samples": latent}
        
        return (image, )    
    
class UL_Image_Generation_Glyph_SDXL_Font:
    @classmethod
    def INPUT_TYPES(s):
        multi_font_idx_list = []
        with open(os.path.join(current_dir, 'Glyph_SDXL_Scripts', 'assets', 'color_idx.json'), 'r') as f:
            color_idx_dict = json.load(f)
            color_idx_list = list(color_idx_dict)
        multilingual_font_dict = {}
        multilingual_meta_path = os.path.join(current_dir, 'Glyph_SDXL_Scripts', 'assets', 'multi_fonts')
        for code in multilingual_code_dict:
            with open(os.path.join(multilingual_meta_path, f"{code}.json"), 'r') as f:
                lang_font_list = json.load(f)
            multilingual_font_dict[code] = lang_font_list
        multi_font_idx_list = []
        for lang in multilingual_font_dict:
            with open(os.path.join(current_dir, 'Glyph_SDXL_Scripts', 'assets', 'multi_fonts', f'{lang}.json'), 'r') as f:
                lang_font_list = json.load(f)
            for font in lang_font_list:
                font_name = font[0][3:]
                multi_font_idx_list.append(f"{multilingual_code_dict[lang]}: {font_name}")
        return {
            "required": {
                "font_text1": ("STRING", {"multiline": True,  "default": "身无彩凤双飞翼，\n心有灵犀一点通。", "dynamicPrompts": True}),
                "font_color1": (color_idx_list, {"default": "red"}),
                "font_name1": (multi_font_idx_list, {"default": "Chinese: HYQiHei-AZEJ"}),
                
                "font_text2": ("STRING", {"multiline": True,  "default": "除夜を祝う.", "dynamicPrompts": True}),
                "font_color2": (color_idx_list, {"default": "black"}),
                "font_name2": (multi_font_idx_list, {"default": "Japanese: JackeyFont"}),
                
                "font_text3": ("STRING", {"multiline": True,  "default": "It's not easy to build a node.", "dynamicPrompts": True}),
                "font_color3": (color_idx_list, {"default": "yellow"}),
                "font_name3": (multi_font_idx_list, {"default": "English: DMSerifDisplay-Regular"}),
                
                "font_text4": ("STRING", {"multiline": True,  "default": "전문 메이크업 아티스트 아름다운 한복 무료 촬영.", "dynamicPrompts": True}),
                "font_color4": (color_idx_list, {"default": "brown"}),
                "font_name4": (multi_font_idx_list, {"default": "Korean: TDTDLatteOTF"}),
                
                "font_text5": ("STRING", {"multiline": True,  "default": "Ответьте, пожалуйста, на номер +123-456-7890.", "dynamicPrompts": True}),
                "font_color5": (color_idx_list, {"default": "gold"}),
                "font_name5": (multi_font_idx_list, {"default": "Russian: TTRamillas-Italic"}),
                
                "font_text6": ("STRING", {"multiline": True,  "default": "A TERRA É O QUE TODOS NÓS TEMOS EM COMUM.", "dynamicPrompts": True}),
                "font_color6": (color_idx_list, {"default": "chocolate"}),
                "font_name6": (multi_font_idx_list, {"default": "Portuguese: Gliker-Regular"}),
                }
            }

    RETURN_TYPES = ("Glyph_SDXL_Font_Params", )
    RETURN_NAMES = ("font_params", )
    FUNCTION = "UL_Image_Generation_Glyph_SDXL_Font"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "Glyph-SDXL Font"
    
    # OUTPUT_NODE = True
    # OUTPUT_IS_LIST = (False,)

    def UL_Image_Generation_Glyph_SDXL_Font(self, 
                                            font_text1, font_color1, font_name1, 
                                            font_text2='', font_color2=None, font_name2=None, 
                                            font_text3='', font_color3=None, font_name3=None,
                                            font_text4='', font_color4=None, font_name4=None,
                                            font_text5='', font_color5=None, font_name5=None,
                                            font_text6='', font_color6=None, font_name6=None,
                                            font_text7='', font_color7=None, font_name7=None,
                                            font_text8='', font_color8=None, font_name8=None,
                                            font_text9='', font_color9=None, font_name9=None,
                                            font_text10='', font_color10=None, font_name10=None,
                                            font_text11='', font_color11=None, font_name11=None,
                                            font_text12='', font_color12=None, font_name12=None,
                                            font_text13='', font_color13=None, font_name13=None,
                                            font_text14='', font_color14=None, font_name14=None,
                                            font_text15='', font_color15=None, font_name15=None,
                                            font_text16='', font_color16=None, font_name16=None,
                                            font_text17='', font_color17=None, font_name17=None,
                                            font_text18='', font_color18=None, font_name18=None,
                                            font_text19='', font_color19=None, font_name19=None,
                                            font_text20='', font_color20=None, font_name20=None,
                                            ):
        # Glyph_SDXL_Font_params = {
        #     "font_text": font_text1,
        #     "font_color": font_color1,
        #     "font_name": font_name1,
        # }
        Glyph_SDXL_conditions = (
            font_text1, font_text2, font_text3, font_text4, font_text5, 
            font_text6, font_text7, font_text8, font_text9, font_text10, 
            font_text11, font_text12, font_text13, font_text14, font_text15, 
            font_text16, font_text17, font_text18, font_text19, font_text20, 
            font_color1, font_color2, font_color3, font_color4, font_color5, 
            font_color6, font_color7, font_color8, font_color9, font_color10, 
            font_color11, font_color12, font_color13, font_color14, font_color15, 
            font_color16, font_color17, font_color18, font_color19, font_color20, 
            font_name1, font_name2, font_name3, font_name4, font_name5, 
            font_name6, font_name7, font_name8, font_name9, font_name10, 
            font_name11, font_name12, font_name13, font_name14, font_name15, 
            font_name16, font_name17, font_name18, font_name19, font_name20
            )
        
        # return (Glyph_SDXL_Font_params, )
        return (Glyph_SDXL_conditions, )
    
class Glyph_SDXL_Checkponits_Loader():
    @classmethod
    def INPUT_TYPES(cls):
        clip_list = os.listdir(os.path.join(folder_paths.models_dir, "text_encoders"))
        clip_folders = [folder for folder in clip_list if os.path.isdir(os.path.join(folder_paths.models_dir, "text_encoders", folder))]
        return {
            "required": {
                "diffusers_model": ("Diffusers_Model", ),
                "clip_name": (["Auto_DownLoad"] + clip_folders, {"tooltip": "If Auto_Download selected, clip model files `google/byt5-small` will cached (Auto_Download_Path not checked) or download into `ComfyUI\models\clip` (Auto_Download_Path checked).\n如果选择自动下载(Auto_DownLoad)且以前没下载过并且勾选(Auto_Download_Path)下载到本地，clip模型文件 `google/byt5-small`将下载到`ComfyUI\models\clip`，否则缓存到huggingface缓存路径。"}),
                "version": ("BOOLEAN", {"default": True, "label_on": "v2", "label_off": "v1", "tooltip": "V2 for 10 languages, v1 just for English."}),
                "Auto_Download_Path": ("BOOLEAN", {"default": True, "label_on": "models_local本地", "label_off": ".cache缓存", "tooltip": "Cache clip model files to huggingface cache_dir or download into `ComfyUI\models\clip`.\nclip模型自动下载位置选择：huggingface缓存路径或者`ComfyUI\models\clip`。"}),
            },
            "optional":{
            },
        }

    RETURN_TYPES = ("Glyph_SDXL_Model",)
    RETURN_NAMES = ("glyph_sdxl_model",)
    FUNCTION = "loader"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "Glyph-SDXL Model Loader"
    DESCRIPTION = "Patch unet and apply unet lora and load byt5 clip (text_encoder).\n给unet作patch并且应用lora并且加载byt5 clip (text_encoder)。"
    
    def __init__(self):
        self.version = None
        self.byt5_model = None
        self.byt5_mapper = None
        self.byt5_tokenizer = None
        self.unet = None

    def loader(self, version, clip_name, Auto_Download_Path, diffusers_model, debug=True):
        from .Glyph_SDXL_Scripts.glyph_sdxl.utils.parse_config import parse_config
        
        color_ann_path = os.path.join(current_dir, 'Glyph_SDXL_Scripts', 'assets', 'color_idx.json')
        
        Glyph_SDXL_checkpoints_dir = os.path.join(current_dir, 'checkpoints', 'glyph-sdxl')
        config = parse_config(os.path.join(current_dir, 'Glyph_SDXL_Scripts', 'configs', 'glyph_sdxl_albedo.py'))
        font_ann_path = os.path.join(current_dir, 'Glyph_SDXL_Scripts', 'assets', 'font_idx_512.json')
        if version: # v2 multilingual
            Glyph_SDXL_checkpoints_dir = os.path.join(current_dir, 'checkpoints', 'glyph-sdxl_multilingual_10-lang')
            config = parse_config(os.path.join(current_dir, 'Glyph_SDXL_Scripts', 'configs', 'glyph_sdxl_multilingual_albedo.py'))
            font_ann_path = os.path.join(current_dir, 'Glyph_SDXL_Scripts', 'assets', 'multilingual_10-lang_idx.json')
            
        if debug:
            print('\033[93m', 'Load Glyph-SDXL v2 multilingual checkpoints.', '\033[0m') if version else print('\033[93m', 'Load Glyph-SDXL v1 english checkpoints.', '\033[0m')
        
        
        clip_path = os.path.join(folder_paths.models_dir, "text_encoders", clip_name)
        if not os.path.exists(clip_path):
            if Auto_Download_Path:
                clip_path = os.path.join(folder_paths.models_dir, "text_encoders", 'models--google--byt5-small')
                if not os.path.exists(os.path.join(clip_path, 'pytorch_model.bin')):
                    download_repoid_model_from_huggingface("google/byt5-small", clip_path, ignore_patterns=[".msgpack", ".h5"])
            else:
                clip_path = "google/byt5-small"
        
        if self.version != version or self.unet != diffusers_model['unet']:
            self.version = version
            self.unet = diffusers_model['unet']
            from .Glyph_SDXL_Scripts.glyph_sdxl.utils.load_pretrained_byt5 import load_byt5_and_byt5_tokenizer
            from .Glyph_SDXL_Scripts.glyph_sdxl.modules import T5EncoderBlockByT5Mapper
            from .Glyph_SDXL_Scripts.glyph_sdxl.custom_diffusers.models import CrossAttnInsertBasicTransformerBlock
            from diffusers.models.attention import BasicTransformerBlock
            from peft import LoraConfig
            from peft.utils import set_peft_model_state_dict
            
            if self.byt5_model == None:
                self.byt5_model, self.byt5_tokenizer = load_byt5_and_byt5_tokenizer(
                    byt5_name=clip_path, 
                    color_ann_path=color_ann_path, 
                    font_ann_path=font_ann_path, 
                    **config.byt5_config)
            
            byt5_mapper_dict = [T5EncoderBlockByT5Mapper]
            byt5_mapper_dict = {mapper.__name__: mapper for mapper in byt5_mapper_dict}
            
            self.byt5_mapper = byt5_mapper_dict[config.byt5_mapper_type](
                self.byt5_model.config,
                **config.byt5_mapper_config,
            )
            
            inserted_new_modules_para_set = set()
            unet_patch_steps = 0
            if debug:
                print('\033[93m', f'Start Unet Patch!', '\033[0m')
            for name, module in diffusers_model['unet'].named_modules():
                if isinstance(module, BasicTransformerBlock) and name in config.attn_block_to_modify:
                    unet_patch_steps += 1
                    parent_module = diffusers_model['unet']
                    for n in name.split(".")[:-1]:
                        parent_module = getattr(parent_module, n)
                    new_block = CrossAttnInsertBasicTransformerBlock.from_transformer_block(
                        module,
                        self.byt5_model.config.d_model if config.byt5_mapper_config.sdxl_channels is None else config.byt5_mapper_config.sdxl_channels,
                    )
                    new_block.requires_grad_(False)
                    for inserted_module_name, inserted_module in zip(
                        new_block.get_inserted_modules_names(), 
                        new_block.get_inserted_modules()
                    ):
                        inserted_module.requires_grad_(True)
                        for para_name, para in inserted_module.named_parameters():
                            para_key = name + '.' + inserted_module_name + '.' + para_name
                            assert para_key not in inserted_new_modules_para_set
                            inserted_new_modules_para_set.add(para_key)
                    for origin_module in new_block.get_origin_modules():
                        origin_module.to(diffusers_model['unet'].dtype)
                    parent_module.register_module(name.split(".")[-1], new_block)
                    print(f"inserted cross attn block to {name}")
                    
            if debug:
                    print('\033[93m', f'Unet Patch Completed! Total blocks: {unet_patch_steps}', '\033[0m')
                
            unet_lora_target_modules = [
                "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
                "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
            ]
            unet_lora_config = LoraConfig(
                r=config.unet_lora_rank,
                lora_alpha=config.unet_lora_rank,
                init_lora_weights="gaussian",
                target_modules=unet_lora_target_modules,
            )
            diffusers_model['unet'].add_adapter(unet_lora_config)
            
            state_dict = load_torch_file(os.path.join(Glyph_SDXL_checkpoints_dir, 'unet_lora.pt'), safe_load=True)
            incompatible_keys = set_peft_model_state_dict(diffusers_model['unet'], state_dict, adapter_name="default")
            if getattr(incompatible_keys, 'unexpected_keys', []) == []:
                print(f"loaded unet_lora_layers_para_multilingual")
            else:
                print(f"unet_lora_layers_multilingual has unexpected_keys: {getattr(incompatible_keys, 'unexpected_keys', None)}")
            
            state_dict = load_torch_file(os.path.join(Glyph_SDXL_checkpoints_dir, 'unet_inserted_attn.pt'), safe_load=True)
            missing_keys, unexpected_keys = diffusers_model['unet'].load_state_dict(state_dict, strict=False)
            assert len(unexpected_keys) == 0, unexpected_keys
            
            state_dict = load_torch_file(os.path.join(Glyph_SDXL_checkpoints_dir, 'byt5_mapper.pt'), safe_load=True)
            self.byt5_mapper.load_state_dict(state_dict)
            
            state_dict = load_torch_file(os.path.join(Glyph_SDXL_checkpoints_dir, 'byt5_model.pt'), safe_load=True)
            self.byt5_model.load_state_dict(state_dict)
            del state_dict
            
        glyph_sdxl_model = {
            'diffusers_model': diffusers_model,
            'byt5_text_encoder': self.byt5_model,
            'byt5_tokenizer': self.byt5_tokenizer,
            'byt5_mapper': self.byt5_mapper,
            'byt5_max_length': config.byt5_max_length,
            'version': version,
        }
        
        return (glyph_sdxl_model, )
    
NODE_CLASS_MAPPINGS = {
    "UL_Image_Generation_Glyph_SDXL": UL_Image_Generation_Glyph_SDXL_Sampler,
    "UL_Image_Generation_Glyph_SDXL_Font": UL_Image_Generation_Glyph_SDXL_Font,
    "UL_Image_Generation_Glyph_SDXL_Model_Loader": Glyph_SDXL_Checkponits_Loader,
}

multilingual_code_dict = {
    'cn': 'Chinese',
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'jp': 'Japanese',
    'kr': 'Korean',
}
multilingual_reverse_code_dict = {
    'Chinese': 'cn',
    'English': 'en',
    'French': 'en',
    'German': 'en',
    'Spanish': 'en',
    'Italian': 'en',
    'Portuguese': 'en',
    'Russian': 'en',
    'Japanese': 'jp',
    'Korean': 'kr',
}