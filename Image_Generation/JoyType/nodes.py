import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from ...UL_common.common import get_files_with_extension, tensor2numpy_cv2, SD15_Scheduler_List, Scheduler_Names, get_device_by_name, pil2tensor, clean_up, tensor2pil, cv2img_canny, seperate_masks, Pillow_Color_Names, padding_image
import cv2
import folder_paths
import torch
from comfy.model_management import vae_offload_device, unet_offload_device, text_encoder_offload_device

class UL_Image_Generation_JoyType_Render_List:
    @classmethod
    def INPUT_TYPES(self):
        self.font_files = get_files_with_extension('fonts', ['.ttf', '.otf'])
        return {
            "required": {
            "mask": ("MASK",),
            "mask_gap": ("INT", {"default": 102,"min": 0, "max": 10240, "step": 1, "tooltip": "Seperate masks from input mask."}),
            "sort_radio": ("BOOLEAN", {"default": True, "label_on": "↔水平排序", "label_off": "↕垂直排序", "tooltip": "控制生成文字的位置顺序，根据遮罩的顺序确认文字对应位置的顺序。水平则从左往右，最靠近画布左边的遮罩位置先开始，垂直则从上往下，最靠近画布上边的遮罩位置开始。"}), 
            # "width": ("INT", {"forceInput": True}),
            # "height": ("INT", {"forceInput": True}),
            "font1": (['None'] + [file for file in self.font_files], {"default": "清松手写体.ttf"}),
            "text1": ("STRING", {"default": "床前明月光，疑似地上霜。", "multiline": True, "dynamicPrompts": True}),
            "font2": (['None'] + [file for file in self.font_files], {"default": "小篆拼音体.ttf"}),
            "text2": ("STRING", {"default": "除夜を祝う.", "multiline": True, "dynamicPrompts": True}),
            "font3": (['None'] + [file for file in self.font_files], {"default": "迷你简综艺.ttf"}),
            "text3": ("STRING", {"default": "It's funny!", "multiline": True, "dynamicPrompts": True}),
            "font4": (['None'] + [file for file in self.font_files], {"default": "沙孟海书法字体.ttf"}),
            "text4": ("STRING", {"default": "전문 메이크업 아티스트 아름다운 한복 무료 촬영.", "multiline": True, "dynamicPrompts": True}),
            "font5": (['None'] + [file for file in self.font_files], {"default": "文道现代篆书.ttf"}),
            "text5": ("STRING", {"default": "Ответьте, пожалуйста, на номер +123-456-7890.", "multiline": True, "dynamicPrompts": True}),
            "font6": (['None'] + [file for file in self.font_files], {"default": "AnyText-Arial-Unicode.ttf"}),
            "text6": ("STRING", {"default": "A TERRA É O QUE TODOS NÓS TEMOS EM COMUM.", "multiline": True, "dynamicPrompts": True}),
        }, 
        }

    RETURN_TYPES = ("JoyType_Params", )
    RETURN_NAMES = ("params", )
    FUNCTION = "UL_Image_Generation_JoyType_Render_List"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "JoyType List"
    DESCRIPTION = ""

    def __init__(self):
        pass

    def UL_Image_Generation_JoyType_Render_List(self, mask, font1, text1, font2=None, text2=None, font3=None, text3=None, font4=None, text4=None, font5=None, text5=None, font6=None, text6=None, font7=None, text7=None, font8=None, text8=None, font9=None, text9=None, font10=None, text10=None, sort_radio=True, mask_gap=102):
        
        cv2_mask = tensor2numpy_cv2(mask)
        size = (cv2_mask.shape[1], cv2_mask.shape[0])
        
        sort_radio = '↔' if sort_radio else '↕'
        seperated_masks = seperate_masks(cv2_mask, sort_radio, mask_gap)
        mask_box_list = []
        for mask in seperated_masks:
            y_coords, x_coords = np.nonzero(mask)
            stack = [x_coords.min(), y_coords.min(), (x_coords.max() - x_coords.min()), (y_coords.max() - y_coords.min())]
            mask_box_list.append(stack)
        
        if sort_radio == '↕': # 根据列表中第几个值大小，由小到大重新排序
            def takeSecond(elem):
                return elem[1] # 第二个值y_min
            mask_box_list.sort(key=takeSecond)
        if sort_radio == '↔':
            def takeSecond(elem):
                return elem[0] # 第一个值，x_min
            mask_box_list.sort(key=takeSecond)
        
        MAX_TEXT_BOX = len(mask_box_list)
            
        fonts = [font1, font2, font3, font4, font5, font6, font7, font8, font9, font10]
        texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10]
        font_paths_list = []
        for font in fonts[:MAX_TEXT_BOX]:
            if font != None:
                font_path = os.path.join(folder_paths.models_dir, "fonts", font)
                font_paths_list.append(font_path)
        text_list = []
        for text in texts[:MAX_TEXT_BOX]:
            if text != None:
                text_list.append(text)
            
        conditions = tuple(text_list + mask_box_list + font_paths_list)
        
        font_text = []
        mask_boxs = []
        font_paths = []
        for i in range(MAX_TEXT_BOX):
            font_text.append(conditions[i])
            mask_boxs.append(conditions[i + MAX_TEXT_BOX])
            font_paths.append(conditions[i + MAX_TEXT_BOX * 2])
            
        render_list = []
        for i, (font_texts, masks, font_file) in enumerate(zip(font_text, mask_boxs, font_paths)):
            render_list.append(
                {
                    "text": font_texts,
                    "polygon": masks,
                    "font_path": font_file,
                }
            )
        
        JoyType_Params = {
            "render_list": render_list, 
            "size": size,
        }
        
        return (JoyType_Params, )

class UL_Image_Generation_JoyType_Font_Img:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
            "params": ("JoyType_Params", ),
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
        }, 
        # "optional": {
        #     "list2": ("JoyType_Render_List", ),
        #     "list3": ("JoyType_Render_List", ),
        #     }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", )
    RETURN_NAMES = ("canny_image", "font_image", )
    FUNCTION = "UL_Image_Generation_JoyType_Font_Img"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "JoyType Font_Img"
    DESCRIPTION = """"""

    def UL_Image_Generation_JoyType_Font_Img(self, font_color_name, font_color_code, font_color_mode, bg_color_name, bg_color_code, bg_color_mode, font_color_codeR, font_color_codeG, font_color_codeB, font_color_codeA, params):
        size = params['size']
        render_list = params['render_list']
        
        if bg_color_mode:
            bg_color = bg_color_name
            if bg_color_name == 'transparent':
                bg_color = (0,0,0,0)
        else:
            bg_color =  "#" + str(bg_color_code).replace("#", "").replace(":", "").replace(" ", "") # 颜色码-绿色：#00FF00
        
        if font_color_mode:
            font_color = font_color_name
            if font_color_name == 'transparent':
                font_color = (0,0,0,0)
        elif not font_color_mode and font_color_codeR == -1:
            font_color = "#" + str(font_color_code).replace("#", "").replace(":", "").replace(" ", "") # 颜色码-绿色：#00FF00
        else:
            font_color = (font_color_codeR, font_color_codeG, font_color_codeB, font_color_codeA)
        
        width, height = size[0], size[1]
        font_img = render_all_text(render_list, width, height, font_color=font_color, bg_color=bg_color)
        render_img = cv2img_canny(np.array(font_img))
        
        render_canny_img = pil2tensor(render_img)
        font_img = pil2tensor(np.array(font_img))
        
        return (render_canny_img,  font_img, )

class UL_Image_Generation_Diffusers_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
                {
                "diffusers_model": ("Diffusers_Model", ),
                "prompt": ("STRING", {"default": "beautiful landscape, many peaks, clear lake, ", "multiline": True}),
                "a_prompt": ("STRING", {"default": "best quality, extremely detailed, 4k, HD, supper legible text, clear text edges, clear strokes, neat writing, no watermarks, ", "multiline": True}),
                "n_prompt": ("STRING", {"default": "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture", "multiline": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 1, "max": 99, "step": 0.1}),
                "scheduler": (["PNDM from pretrained"] + Scheduler_Names, {"default": "Euler a"}),
                "width": ("INT", {"default": 256,"min": 512, "max": 10240, "step": 1}),
                "height": ("INT", {"default": 256,"min": 512, "max": 10240, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 960}),
                "seed": ("INT", {"default": 88888888, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Warning: do not delete model unless this node no longer needed, it will try release device_memory and ram. if checked and want to continue node generation, use ComfyUI-Manager `Free model and node cache` to reset node state or change parameter in Loader node to activate.\n注意：仅在这个节点不再需要时删除模型，将尽量释放系统内存和设备专用内存。如果删除后想继续使用此节点，使用ComfyUI-Manager插件的`Free model and node cache`重置节点状态或者更换模型加载节点的参数来激活。"}),
                "keep_model_device": ("BOOLEAN", {"default": True, "label_on": "comfy", "label_off": "device", "tooltip": "Keep model in comfy_auto_unet_offload_device (HIGH_VRAM: device, Others: cpu) or device_memory after generation.\n生图完成后，模型转移到comfy自动选择设备(HIGH_VRAM: device, 其他: cpu)或者保留在设备专用内存上。"}),
                # "sequential_offload": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "off"}),
                },
            "optional": {
                "diffusers_control": ("Diffusers_Control", ),
                },
            }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "sampler"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "JoyType Sampler"
    DESCRIPTION = "Diffusers sd1.5 sampler, control if optional.\ndiffusers的sd1.5采样器，控制网是可选项。"

    def sampler(self, diffusers_model, prompt, a_prompt, n_prompt, steps, cfg, scheduler, seed, device, keep_model_loaded, batch_size, keep_model_device, width, height, diffusers_control=None):
        device = get_device_by_name(device)
        dtype = diffusers_model['unet'].dtype
            
        from diffusers import PNDMScheduler
        pipe_scheduler = PNDMScheduler.from_pretrained(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'JoyType-scheduler'))
        if scheduler != 'PNDM from pretrained':
            pipe_scheduler = SD15_Scheduler_List(pipe_scheduler)[scheduler]
            
        if 'cpu' in diffusers_model['vae'].device.type:
            diffusers_model['vae'].to(device)
        if 'cpu' in diffusers_model['unet'].device.type:
            diffusers_model['unet'].to(device)
        if 'cpu' in diffusers_model['clip']['text_encoder'].device.type:
            diffusers_model['clip']['text_encoder'].to(device)
        from .JoyType_Scripts.JoyType_pipeline import JoyType_pipeline
        from transformers import CLIPImageProcessor
        if diffusers_control != None:
            controlnet_model = diffusers_control['diffusers_control_net']['control_model'].to(device, dtype)
            render_img = tensor2pil(diffusers_control['image'])
            render_img = padding_image(render_img, max(width, height)) # 可否使用vae缩放latent
            width, height = render_img.size
        else:
            controlnet_model = None
        pipe = JoyType_pipeline(
            vae=diffusers_model['vae'], 
            text_encoder=diffusers_model['clip']['text_encoder'], 
            tokenizer=diffusers_model['clip']['tokenizer'], 
            unet=diffusers_model['unet'],
            controlnet=controlnet_model,
            scheduler=pipe_scheduler,
            feature_extractor=CLIPImageProcessor(),
            )
        
        pipe.enable_xformers_memory_efficient_attention()
        # JoyType_pipe.enable_model_cpu_offload()
        # if sequential_offload:
        #     self.JoyType_pipe.enable_sequential_cpu_offload()
        # from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
        # JoyType_pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        
        if diffusers_control != None:
            batch_render_img = [render_img for _ in range(batch_size)]
            control_strength = diffusers_control['strength']
            control_start_percent = diffusers_control['start_percent']
            control_end_percent = diffusers_control['end_percent']
        else:
            batch_render_img = None
            control_strength = None
            control_start_percent = None
            control_end_percent = None
        batch_prompt = [f'{prompt}, {a_prompt}' for _ in range(batch_size)]
        batch_n_prompt = [n_prompt for _ in range(batch_size)]
        
        images = pipe(
            batch_prompt,
            negative_prompt=batch_n_prompt,
            image=batch_render_img,
            controlnet_conditioning_scale=control_strength,
            guidance_scale=cfg,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.manual_seed(seed),
            control_guidance_start=control_start_percent,
            control_guidance_end=control_end_percent,
        ).images
        
        result = []
        for img in images:
            image = pil2tensor(img)
            result.append(image)
        result = torch.cat(result, dim=0)
        
        if keep_model_loaded:
            if keep_model_device:
                diffusers_model['vae'].to(vae_offload_device())
                diffusers_model['unet'].to(unet_offload_device())
                diffusers_model['clip']['text_encoder'].to(text_encoder_offload_device())
                if diffusers_control != None:
                    diffusers_control['diffusers_control_net']['control_model'].to(text_encoder_offload_device())
                clean_up()
        else:
            del pipe.vae
            del pipe.text_encoder
            del pipe.tokenizer
            del pipe.unet
            del pipe.scheduler
            if diffusers_control != None:
                del pipe.controlnet
            del diffusers_model['vae']
            del diffusers_model['clip']
            del diffusers_model['unet']
            del diffusers_model['scheduler']
            del diffusers_model['pipe']
            if diffusers_control != None:
                del diffusers_control['diffusers_control_net']['control_model']
            clean_up()
        
        return (result, )

NODE_CLASS_MAPPINGS = {
    "UL_Image_Generation_JoyType_Render_List": UL_Image_Generation_JoyType_Render_List,
    "UL_Image_Generation_JoyType_Font_Img": UL_Image_Generation_JoyType_Font_Img, 
    "UL_Image_Generation_Diffusers_Sampler": UL_Image_Generation_Diffusers_Sampler,
}

def render_all_text(render_list, width, height, threshold=512, font_color='white', bg_color=(0,0,0,0)):  # 增加字体颜色可选，默认为白色。
# def render_all_text(bg_img: None, text: str, polygon: tuple[int, int, int, int], width: int, height: int, font, threshold=512):
    # print('\033[93m', 'Function: render_all_text', '\n', render_list, '\n', type(render_list), '\033[0m')
    MAX_LENGTH = 99 # 允许的最大字数，超过的字会被忽略。
    # if bg_img == None: # 初始化背景
    board = Image.new('RGBA', (width, height), bg_color) # 最终图的底图，颜色任意，只是个黑板。
    # else:
    #     board = bg_img # 已经生成了背景，利用返回的背景再叠加新生成的字体图。

    for text_dict in render_list: # [{'text': '冬至', 'polygon': [18(最小x到左边的距离，mask左边框到画布左边的距离), 489(总高度减去mask高度和最小y，mask上边框到画布上边的距离), 251(mask宽度), 22(mask高度)], 'font_name': '清松手写体'}]
        text = text_dict["text"]
        polygon = text_dict["polygon"]
        font_path = text_dict["font_path"]
        if len(text) > MAX_LENGTH:
            text = text[:MAX_LENGTH]
            raise ValueError(f'{text}... exceeds the maximum length {MAX_LENGTH} and has been cropped.')

        w, h = polygon[2:]
        vert = True if w < h else False
        image4ratio = Image.new('RGBA', (1024, 1024), (0,0,0,0)) # 字体底图，会贴到最终图底图上，改成透明底图就不会出现遮挡。
        draw = ImageDraw.Draw(image4ratio)

        font = ImageFont.truetype(font_path, encoding='utf-8', size=50)
        # try:
        #     font = ImageFont.truetype(font_path, encoding='utf-8', size=50)
        # except FileNotFoundError:
        #     font = ImageFont.truetype(font_path, encoding='utf-8', size=50)

        if not vert:
            # draw.text(xy=(0, 0), text=text, font=font, fill='white')
            draw.text(xy=(0, 0), text=text, font=font, fill=font_color) # 字体颜色
            _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
            _th += 1
        else:
            _tw, y_c = 0, 0
            for c in text:
                # draw.text(xy=(0, y_c), text=c, font=font, fill='white')
                draw.text(xy=(0, y_c), text=c, font=font, fill=font_color) # 字体颜色
                _l, _t, _r, _b = font.getbbox(c)
                _tw = max(_tw, _r - _l)
                y_c += _b
            _th = y_c + 1

        ratio = (_th * w) / (_tw * h)
        text_img = image4ratio.crop((0, 0, _tw, _th))
        x_offset, y_offset = 0, 0
        if 0.8 <= ratio <= 1.2:
            text_img = text_img.resize((w, h))
        elif ratio < 0.75:
            resize_h = int(_th * (w / _tw))
            text_img = text_img.resize((w, resize_h))
            y_offset = (h - resize_h) // 2
        else:
            resize_w = int(_tw * (h / _th))
            text_img = text_img.resize((resize_w, h))
            x_offset = (w - resize_w) // 2

        r, g, b, a = text_img.split() # 获取要贴进底图的图片的alpha透明通道
        
        board.paste(text_img, (polygon[0] + x_offset, polygon[1] + y_offset), mask=a) # 将透明通道作为遮罩，这样只将字体部分非alpha像素贴上去。

    return board

def resize_w(w, img):
    return cv2.resize(img, (w, img.shape[0]))


def resize_h(h, img):
    return cv2.resize(img, (img.shape[1], h))