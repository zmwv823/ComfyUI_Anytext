# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import torch
# import re
import numpy as np
import cv2
# import einops
# import time

# from .cldm.ddim_hacked import DDIMSampler
from .AnyText_t3_dataset import draw_glyph, draw_glyph2
# from .AnyText_pipeline_util import check_channels, resize_image
# from .AnyText_bert_tokenizer import BasicTokenizer
# import folder_paths
# from comfy.utils import ProgressBar
# from comfy.model_management import soft_empty_cache
# from ...UL_common.common import clean_up

# checker = BasicTokenizer()
# BBOX_MAX_NUM = 20
# PLACE_HOLDER = '*'
# max_chars = 50

# comfyui_models_dir = folder_paths.models_dir
# class AnyText_Pipeline():
#     def __init__(self, anytext_model, loaded_prompt, processed_prompt, texts, translator, use_translator, model_device, model_dtype, translator_device, keep_translator_loaded, keep_translator_device, Auto_Download_Path):
#         self.device = model_device
#         self.dtype = model_dtype
#         self.use_translator = use_translator
#         self.translator_path = translator
#         self.translator_device = translator_device
#         self.keep_translator_loaded = keep_translator_loaded
#         self.keep_translator_device = keep_translator_device
#         self.Auto_Download_Path = Auto_Download_Path
#         self.anytext_model = anytext_model
        
#         self.loaded_t5_translate_en_ru_zh_model_name = None
        
#         if self.use_translator == True:
#             if translator == 'damo/nlp_csanmt_translation_zh2en':
#                 self.trans_pipe = 'damo/nlp_csanmt_translation_zh2en'
#             elif translator == "utrobinmv/t5_translate_en_ru_zh_base_200":
#                 self.trans_pipe = 'utrobinmv/t5_translate_en_ru_zh_base_200'
#             elif translator == "utrobinmv/t5_translate_en_ru_zh_large_1024":
#                 self.trans_pipe = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
#             elif translator == "utrobinmv/t5_translate_en_ru_zh_small_1024":
#                 self.trans_pipe = 'utrobinmv/t5_translate_en_ru_zh_small_1024'
#         else:
#             self.trans_pipe = None
            
#         self.processed_prompt = processed_prompt
#         self.texts = texts
#         self.loaded_prompt = loaded_prompt
    
#     def __call__(self, font, input_tensor, **forward_params):
#         tic = time.time()
        
#         # if 'cpu' in self.anytext_model.device.type:
#         #     self.anytext_model.to(self.device) 
            
#         ddim_sampler = DDIMSampler(self.anytext_model, device=self.device)
        
#         str_warning = ''
#         # get inputs
#         seed = input_tensor.get('seed', -1)
#         self.prompt = input_tensor.get('prompt')
#         draw_pos = input_tensor.get('draw_pos')
#         ori_image = input_tensor.get('ori_image')

#         mode = forward_params.get('mode')
#         translator_device = forward_params.get('translator_device')
#         Random_Gen = forward_params.get('Random_Gen')
#         sort_priority = forward_params.get('sort_priority', '↕')
#         show_debug = forward_params.get('show_debug', False)
#         revise_pos = forward_params.get('revise_pos', False)
#         img_count = forward_params.get('image_count', 1)
#         ddim_steps = forward_params.get('ddim_steps', 20)
#         w = forward_params.get('image_width', 512)
#         h = forward_params.get('image_height', 512)
#         strength = forward_params.get('strength', 1.0)
#         cfg_scale = forward_params.get('cfg_scale', 9.0)
#         eta = forward_params.get('eta', 0.0)
#         a_prompt = forward_params.get('a_prompt', 'best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks')
#         n_prompt = forward_params.get('n_prompt', 'low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture')
        
#         if self.loaded_prompt == None or self.loaded_prompt != self.prompt or self.processed_prompt == None or self.texts == None: # 如果prompt未改变，则不进行prompt处理，防止反复调用翻译。
#             anytext_prompt, texts = self.modify_prompt(self.prompt, translator_device)
#             self.processed_prompt = anytext_prompt
#             self.texts = texts
#         anytext_prompt =self.processed_prompt
#         texts = self.texts
#         self.loaded_prompt = self.prompt
        
#         if anytext_prompt is None and texts is None:
#             return None, -1, "You have input Chinese prompt but the translator is not loaded!", ""
#         n_lines = len(texts)
#         if mode in ['text-generation', 'gen']:
#             if Random_Gen == True:
#                 edit_image = np.ones((h, w, 3)) * 127.5  # empty mask image
#                 edit_image = resize_image(edit_image, max_length=max(h, w))
#                 h, w = edit_image.shape[:2]
#             else:
#                 # edit_image = cv2.imread(draw_pos)[..., ::-1]
#                 edit_image = draw_pos[..., ::-1]
#                 edit_image = resize_image(edit_image, max_length=max(h, w))
#                 h, w = edit_image.shape[:2]
#                 edit_image = np.ones((h, w, 3)) * 127.5  # empty mask image
#         elif mode in ['text-editing', 'edit']:
#             if draw_pos is None or ori_image is None:
#                 return None, -1, "Reference image and position image are needed for text editing!", ""
#             # if isinstance(ori_image, str):
#             if isinstance(ori_image, np.ndarray):
#                 # ori_image = cv2.imread(ori_image)[..., ::-1]
#                 ori_image = ori_image[..., ::-1]
#                 assert ori_image is not None, f"Can't read ori_image image from{ori_image}!"
#             elif isinstance(ori_image, torch.Tensor):
#                 ori_image = ori_image.cpu().numpy()
#             # else:
#             #     assert isinstance(ori_image, np.ndarray), f'Unknown format of ori_image: {type(ori_image)}'
#             edit_image = ori_image.clip(1, 255)  # for mask reason
#             edit_image = check_channels(edit_image)
#             edit_image = resize_image(edit_image, max_length=max(h, w))  # make w h multiple of 64, resize if w or h > max_length
#             h, w = edit_image.shape[:2]  # change h, w by input ref_img
#         # preprocess pos_imgs(if numpy, make sure it's white pos in black bg)
#         if draw_pos is None:
#             pos_imgs = np.zeros((w, h, 1))
#         # if isinstance(draw_pos, str):
#         if isinstance(draw_pos, np.ndarray):
#             # draw_pos = cv2.imread(draw_pos)[..., ::-1]
#             draw_pos = draw_pos[..., ::-1]
#             draw_pos = resize_image(draw_pos, max_length=max(h, w))
#             draw_pos = cv2.resize(draw_pos, (w, h))
#             assert draw_pos is not None, f"Can't read draw_pos image from{draw_pos}!"
#             pos_imgs = 255-draw_pos
#         elif isinstance(draw_pos, torch.Tensor):
#             pos_imgs = draw_pos.cpu().numpy()
#         # else:
#         #     assert isinstance(draw_pos, np.ndarray), f'Unknown format of draw_pos: {type(draw_pos)}'
        
#         pos_imgs = pos_imgs[..., 0:1]
#         pos_imgs = cv2.convertScaleAbs(pos_imgs)
#         _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
#         # seprate pos_imgs
#         pos_imgs = separate_pos_imgs(pos_imgs, sort_priority)
#         if len(pos_imgs) == 0:
#             pos_imgs = [np.zeros((h, w, 1))]
#         if len(pos_imgs) < n_lines:
#             if n_lines == 1 and texts[0] == ' ':
#                 # pass  # text-to-image without text
#                 print('\033[93m', f'Warning: text-to-image without text.', '\033[0m')
#             else:
#                 # return None, -1, f'Found {len(pos_imgs)} positions that < needed {n_lines} from prompt, check and try again!', ''
#                 raise ValueError(f'Found {len(pos_imgs)} positions that < needed {n_lines} from prompt, check and try again(手绘遮罩数少于要绘制的文本数，检查修改再重试)!')
#         elif len(pos_imgs) > n_lines:
#             # str_warning = f'Warning: found {len(pos_imgs)} positions that > needed {n_lines} from prompt.'
#             print('\033[93m', f'Warning: found {len(pos_imgs)} positions that > needed {n_lines} from prompt.', '\033[0m')
#         # get pre_pos, poly_list, hint that needed for anytext
#         pre_pos = []
#         poly_list = []
#         for input_pos in pos_imgs:
#             if input_pos.mean() != 0:
#                 input_pos = input_pos[..., np.newaxis] if len(input_pos.shape) == 2 else input_pos
#                 poly, pos_img = find_polygon(input_pos)
#                 pre_pos += [pos_img/255.]
#                 poly_list += [poly]
#             else:
#                 pre_pos += [np.zeros((h, w, 1))]
#                 poly_list += [None]
#         np_hint = np.sum(pre_pos, axis=0).clip(0, 1)
#         # prepare info dict
#         info = {}
#         info['glyphs'] = []
#         info['gly_line'] = []
#         info['positions'] = []
#         info['n_lines'] = [len(texts)]*img_count
#         gly_pos_imgs = []
#         for i in range(len(texts)):
#             text = texts[i]
#             if len(text) > max_chars:
#                 str_warning = f'"{text}" length > max_chars: {max_chars}, will be cut off...'
#                 text = text[:max_chars]
#             gly_scale = 2
#             if pre_pos[i].mean() != 0:
#                 gly_line = draw_glyph(font, text)
#                 glyphs, _ = draw_glyph2(font, text, poly_list[i], scale=gly_scale, width=w, height=h, add_space=False)
#                 gly_pos_img = cv2.drawContours(glyphs*255, [poly_list[i]*gly_scale], 0, (255, 255, 255), 1)
#                 if revise_pos:
#                     resize_gly = cv2.resize(glyphs, (pre_pos[i].shape[1], pre_pos[i].shape[0]))
#                     new_pos = cv2.morphologyEx((resize_gly*255).astype(np.uint8), cv2.MORPH_CLOSE, kernel=np.ones((resize_gly.shape[0]//10, resize_gly.shape[1]//10), dtype=np.uint8), iterations=1)
#                     new_pos = new_pos[..., np.newaxis] if len(new_pos.shape) == 2 else new_pos
#                     contours, _ = cv2.findContours(new_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#                     if len(contours) != 1:
#                         str_warning = f'Fail to revise position {i} to bounding rect, remain position unchanged...'
#                     else:
#                         rect = cv2.minAreaRect(contours[0])
#                         poly = np.int0(cv2.boxPoints(rect))
#                         pre_pos[i] = cv2.drawContours(new_pos, [poly], -1, 255, -1) / 255.
#                         gly_pos_img = cv2.drawContours(glyphs*255, [poly*gly_scale], 0, (255, 255, 255), 1)
#                 gly_pos_imgs += [gly_pos_img]  # for show
#             else:
#                 glyphs = np.zeros((h*gly_scale, w*gly_scale, 1))
#                 gly_line = np.zeros((80, 512, 1))
#                 gly_pos_imgs += [np.zeros((h*gly_scale, w*gly_scale, 1))]  # for show
#             pos = pre_pos[i]
#             info['glyphs'] += [self.arr2tensor(glyphs, img_count, self.dtype)]
#             info['gly_line'] += [self.arr2tensor(gly_line, img_count, self.dtype)]
#             info['positions'] += [self.arr2tensor(pos, img_count, self.dtype)]
#         # get masked_x
#         masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0)*(1-np_hint)
#         masked_img = np.transpose(masked_img, (2, 0, 1))
#         masked_img = torch.from_numpy(masked_img.copy()).float().to(self.device, self.dtype)
        
#         encoder_posterior = self.anytext_model.encode_first_stage(masked_img[None, ...])
#         masked_x = (self.anytext_model.get_first_stage_encoding(encoder_posterior).detach()).to(self.dtype) #masked latent
        
#         info['masked_x'] = torch.cat([masked_x for _ in range(img_count)], dim=0)

#         hint = self.arr2tensor(np_hint, img_count, self.dtype)
        
#         cond = self.anytext_model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[anytext_prompt + ' , ' + a_prompt] * img_count], text_info=info))
#         un_cond = self.anytext_model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[n_prompt] * img_count], text_info=info))
#         shape = (4, h // 8, w // 8)
#         self.anytext_model.control_scales = ([strength] * 13)
        
#         self.anytext_model.embedding_manager.to('cpu')
#         self.anytext_model.cond_stage_model.to('cpu')
#         self.anytext_model.first_stage_model.to('cpu')
#         soft_empty_cache(True)
#         clean_up()
#         self.anytext_model.model.diffusion_model.to(self.device)
#         self.anytext_model.control_model.to(self.device)
        
#         pbar = ProgressBar(int(ddim_steps))
#         def callback(*_):
#             pbar.update(1)
#         from pytorch_lightning import seed_everything
#         seed_everything(seed)
#         samples, intermediates = ddim_sampler.sample(ddim_steps, img_count,
#                                                           shape, cond, verbose=False, eta=eta,
#                                                           unconditional_guidance_scale=cfg_scale,
#                                                           unconditional_conditioning=un_cond,
#                                                           callback=callback, # 后端代码有timesteps的callback
#                                                           )
        
#         self.anytext_model.model.diffusion_model.to('cpu')
#         self.anytext_model.control_model.to('cpu')
#         soft_empty_cache(True)
#         clean_up()
#         self.anytext_model.first_stage_model.to(self.device)
        
#         x_samples = self.anytext_model.decode_first_stage(samples.to(self.dtype))
#         self.anytext_model.first_stage_model.to('cpu')
#         soft_empty_cache(True)
#         clean_up()
#         x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
#         results = [x_samples[i] for i in range(img_count)]
#         # if mode == 'edit' and False:  # replace backgound in text editing but not ideal yet
#         #     results = [r*np_hint+edit_image*(1-np_hint) for r in results]
#         #     results = [r.clip(0, 255).astype(np.uint8) for r in results]
#         if len(gly_pos_imgs) > 0 and show_debug:
#             glyph_bs = np.stack(gly_pos_imgs, axis=2)
#             glyph_img = np.sum(glyph_bs, axis=2) * 255
#             glyph_img = glyph_img.clip(0, 255).astype(np.uint8)
#             results += [np.repeat(glyph_img, 3, axis=2)]
#         input_prompt = anytext_prompt
#         for t in texts:
#             input_prompt = input_prompt.replace('*', f'"{t}"', 1)
#         print(f'Prompt: {input_prompt}')
#         # debug_info
#         if not show_debug:
#             debug_info = ''
#         else:
#             debug_info = f'\033[93mPrompt(提示词): {input_prompt}\n\033[0m \
#                            \033[93mSize(尺寸): {w}x{h}\n\033[0m \
#                            \033[93mImage Count(生成数量): {img_count}\n\033[0m \
#                            \033[93mSeed(种子): {seed}\n\033[0m \
#                            \033[93mModel Precision(模型精度): {self.dtype}\n\033[0m \
#                            \033[93mCost Time(生成耗时): {(time.time()-tic):.2f}s\033[0m'
#         rst_code = 1 if str_warning else 0
        
#         return (x_samples, results, rst_code, str_warning, debug_info, self.processed_prompt, self.texts, self.loaded_prompt, )
    
#     def modify_prompt(self, prompt, device):
#         prompt = prompt.replace('“', '"')
#         prompt = prompt.replace('”', '"')
#         p = '"(.*?)"'
#         strs = re.findall(p, prompt)
#         if len(strs) == 0:
#             strs = [' ']
#         else:
#             for s in strs:
#                 prompt = prompt.replace(f'"{s}"', f' {PLACE_HOLDER} ', 1)
#         if self.is_chinese(prompt):
#             if self.trans_pipe is None:
#                 return None, None
#             old_prompt = prompt
            
#             from ...Data_Process.utils import t5_translate_en_ru_zh, nlp_csanmt_translation_zh2en
#             self.t5_translate_tensor = None
            
#             if self.trans_pipe == 'utrobinmv/t5_translate_en_ru_zh_small_1024':
#                 zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_small_1024")
#                 if not os.path.exists(os.path.join(zh2en_path, "model.safetensors")) and not self.Auto_Download_Path:
#                     zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_small_1024'
#                 prompt, self.t5_translate_tensor, self.loaded_t5_translate_en_ru_zh_mdoel_name = t5_translate_en_ru_zh('en', prompt + ' .', zh2en_path, self.translator_device, self.keep_translator_loaded, self.t5_translate_tensor, self.loaded_t5_translate_en_ru_zh_model_name, keep_model_device=self.keep_translator_device)
#                 prompt = prompt[0]
                
#             elif self.trans_pipe == 'utrobinmv/t5_translate_en_ru_zh_base_200':
#                 zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_base_200")
#                 if not os.path.exists(os.path.join(zh2en_path, "model.safetensors")) and not self.Auto_Download_Path:
#                     zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_base_200'
#                 prompt, self.t5_translate_tensor, self.loaded_t5_translate_en_ru_zh_mdoel_name = t5_translate_en_ru_zh('en', prompt + ' .', zh2en_path, self.translator_device, self.keep_translator_loaded, self.t5_translate_tensor, self.loaded_t5_translate_en_ru_zh_model_name, keep_model_device=self.keep_translator_device)
#                 prompt = prompt[0]
                
#             elif self.trans_pipe == 'utrobinmv/t5_translate_en_ru_zh_large_1024':
#                 zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_large_1024")
#                 if not os.path.exists(os.path.join(zh2en_path, "model.safetensors")) and not self.Auto_Download_Path:
#                     zh2en_path = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
#                 prompt, self.t5_translate_tensor, self.loaded_t5_translate_en_ru_zh_mdoel_name = t5_translate_en_ru_zh('en', prompt + ' .', zh2en_path, self.translator_device, self.keep_translator_loaded, self.t5_translate_tensor, self.loaded_t5_translate_en_ru_zh_model_name, keep_model_device=self.keep_translator_device)
#                 prompt = prompt[0]
                
#             else:
#                 nlp_device = 'gpu'
#                 if 'cpu' in device.type:
#                     nlp_device = 'cpu'
#                 zh2en_path = os.path.join(comfyui_models_dir, 'prompt_generator', 'modelscope--damo--nlp_csanmt_translation_zh2en')
#                 if not os.path.exists(os.path.join(zh2en_path, "tf_ckpts", "ckpt-0.data-00000-of-00001")) and not self.Auto_Download_Path:
#                     zh2en_path = "damo/nlp_csanmt_translation_zh2en"
#                 prompt = nlp_csanmt_translation_zh2en(nlp_device, prompt + ' .', zh2en_path)['translation']
#             print(f'Translate: {old_prompt} --> {prompt}')
#         return prompt, strs

#     def is_chinese(self, text):
#         text = checker._clean_text(text)
#         for char in text:
#             cp = ord(char)
#             if checker._is_chinese_char(cp):
#                 return True
#         return False

#     def arr2tensor(self, arr, bs, dtype):
#         arr = np.transpose(arr, (2, 0, 1))
#         _arr = torch.from_numpy(arr.copy()).float().to(self.device)
#         _arr = (torch.stack([_arr for _ in range(bs)], dim=0)).to(dtype)
#         return _arr
    
def separate_pos_imgs(img, sort_priority, gap=102):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    components = []
    for label in range(1, num_labels):
        component = np.zeros_like(img)
        component[labels == label] = 255
        components.append((component, centroids[label]))
    if sort_priority == '↕':
        fir, sec = 1, 0  # top-down first
    elif sort_priority == '↔':
        fir, sec = 0, 1  # left-right first
    components.sort(key=lambda c: (c[1][fir]//gap, c[1][sec]//gap))
    sorted_components = [c[0] for c in components]
    return sorted_components

def find_polygon(image, min_rect=False):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=cv2.contourArea)  # get contour with max area
    if min_rect:
        # get minimum enclosing rectangle
        rect = cv2.minAreaRect(max_contour)
        poly = np.int0(cv2.boxPoints(rect))
    else:
        # get approximate polygon
        epsilon = 0.01 * cv2.arcLength(max_contour, True)
        poly = cv2.approxPolyDP(max_contour, epsilon, True)
        n, _, xy = poly.shape
        poly = poly.reshape(n, xy)
    cv2.drawContours(image, [poly], -1, 255, -1)
    return poly, image