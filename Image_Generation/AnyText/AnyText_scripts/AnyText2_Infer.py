import cv2
import torch
import numpy as np
from .AnyText2_utils import check_channels, resize_image, separate_pos_imgs, find_polygon, arr2tensor
from .AnyText2_t3_dataset import draw_glyph, draw_glyph2, draw_font_hint
from AnyText2ControlDiffusion.cldm.recognizer import crop_image
from ....UL_common.common import tensor2pil
import os
from .AnyText_Infer import generate_rectangles

mimic_list = font_hint_image = font_hint_mask =[]

def encode(
    model, font, fonts, mask, prompt, texts, latent, mode, sort_radio, a_prompt, n_prompt, revise_pos, random_mask, Random_Gen_Mask_path, image, font_apply, show_glyph, **extra_kwargs
):
    max_chars = 50
    batch_size, height, width = latent["samples"].shape[0], latent["samples"].shape[2] * 8, latent["samples"].shape[3] * 8 # B, C, H, W
    sort_radio = '↔' if sort_radio else '↕'
    
    img_count = batch_size
    h, w = height, width
    glyline_font_path, text_colors = fonts.glyline_font_path, fonts.text_colors
    n_lines = len(texts)
    
    if not random_mask:
        pos_image = np.array(tensor2pil(mask))
        pos_image = 255 - resize_image(pos_image, max(height, width))#[..., :3]
    else:
        generate_rectangles(width, height, n_lines, 500, Random_Gen_Mask_path)
        pos_image = cv2.imread(Random_Gen_Mask_path)[..., ::-1]
    if random_mask:
        pos_image = pos_image[..., 0:1]
    draw_pos = pos_image
    
    h, w = draw_pos.shape[:2]
    
    if mode:
        mode = 'text-generation'
        revise_pos = revise_pos
    else:
        if image == None:
            raise ValueError('Edit mode need a image.')
        ori_image = np.array(tensor2pil(image))
        mode = 'text-editing'
        revise_pos = False
    
    if mode in ['text-generation', 'gen']:
        edit_image = np.zeros((h, w, 3))
    elif mode in ['text-editing', 'edit']:
        if draw_pos is None or ori_image is None:
            return None, -1, "Reference image and position image are needed for text editing!", ""
        if isinstance(ori_image, str):
            ori_image = cv2.imread(ori_image)[..., ::-1]
            assert ori_image is not None, f"Can't read ori_image image from{ori_image}!"
        elif isinstance(ori_image, torch.Tensor):
            ori_image = ori_image.cpu().numpy()
        else:
            assert isinstance(ori_image, np.ndarray), f'Unknown format of ori_image: {type(ori_image)}'
        # edit_image = ori_image[..., ::-1] #bgr2rgb
        edit_image = ori_image.clip(1, 255)  # for mask reason
        edit_image = check_channels(edit_image)
        edit_image = resize_image(edit_image, max_length=max(height, width))  # make w h multiple of 64, resize if w or h > max_length
        h, w = edit_image.shape[:2]  # change h, w by input ref_img
        
    # preprocess pos_imgs(if numpy, make sure it's white pos in black bg)
    if draw_pos is None:
        pos_imgs = np.zeros((w, h, 1))
    if isinstance(draw_pos, str):
        draw_pos = cv2.imread(draw_pos)[..., ::-1]
        assert draw_pos is not None, f"Can't read draw_pos image from{draw_pos}!"
        pos_imgs = 255-draw_pos
    elif isinstance(draw_pos, torch.Tensor):
        pos_imgs = draw_pos.cpu().numpy()
    else:
        # assert isinstance(draw_pos, np.ndarray), f'Unknown format of draw_pos: {type(draw_pos)}'
        pos_imgs = cv2.resize(draw_pos, (w, h))
    if mode in ['text-editing', 'edit']:
        pos_imgs = cv2.resize(pos_imgs, (w, h))
        
    pos_imgs = 255 - pos_imgs#[..., 0:1]
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
    info['n_lines'] = [len(texts)]*img_count
    font_hint = []
    font_paths = ['None' for i in range(len(texts))]
    if glyline_font_path:
        glyline_font_path = glyline_font_path[:len(texts)]
        font_paths[:len(glyline_font_path)] = glyline_font_path
    info['colors'] = [np.array([500, 500, 500]) for i in range(len(texts))]
    if text_colors:
        text_colors = text_colors.strip().split()[:len(texts)]
        info['colors'][:len(text_colors)] = [np.array([int(p) for p in s.split(',')]) for s in text_colors]

    gly_pos_imgs = []
    font_hint_mimic_imgs = []
    
    gly_line_list = []
    glyphs_list = []
    
    for i in range(len(texts)):
        text = texts[i]
        if len(text) > max_chars:
            # str_warning = f'"{text}" length > max_chars: {max_chars}, will be cut off...'
            print(f'"{text}" length > max_chars: {max_chars}, will be cut off...')
            text = text[:max_chars]
            
        gly_scale = model.model.control_model.glyph_scale
        
        if pre_pos[i].mean() != 0:
            if font_apply:
                gly_line = draw_glyph(font, text, info['colors'][i], show_glyph)
                glyphs = draw_glyph2(font, text, poly_list[i], info['colors'][i], scale=gly_scale, width=w, height=h, add_space=True, show_glyph=show_glyph)
            else:
                gly_line = draw_glyph(font_paths[i], text, info['colors'][i], show_glyph)
                glyphs = draw_glyph2(font_paths[i], text, poly_list[i], info['colors'][i], scale=gly_scale, width=w, height=h, add_space=True, show_glyph=show_glyph)
            
            gly_line_list+=[gly_line]
            glyphs_list+=[glyphs]

            if i < len(font_hint_image) and font_hint_image[i] is not None:
                hint_poly = font_hint_mask[i]
                poly, _ = find_polygon(hint_poly)
                font_hint_mimic_img, _ = draw_font_hint((font_hint_image[i]/127.5-1), poly)
                font_hint_mimic_img = torch.from_numpy(font_hint_mimic_img*255).permute(2, 0, 1).repeat(3, 1, 1)
                font_hint_mimic_imgs += [crop_image(font_hint_mimic_img, hint_poly)]  # chw, tensor, 0-255
                font_paths[i] = 'None'  # not render
            else:
                font_hint_mimic_imgs += [None]
            if font_apply:
                font_hint_line = draw_glyph2(font_paths[i], text, poly_list[i], np.array([255, 255, 255]), scale=1, width=w, height=h, add_space=True, show_glyph=show_glyph)
            else:
                font_hint_line = draw_glyph2(font_paths[i], text, poly_list[i], info['colors'][i], scale=1, width=w, height=h, add_space=True, show_glyph=show_glyph)
            gly_pos_img = cv2.drawContours(glyphs*255, [poly_list[i]*gly_scale], 0, (255, 255, 255), 1)
            
            if revise_pos:
                resize_gly = cv2.resize(glyphs, (pre_pos[i].shape[1], pre_pos[i].shape[0]))
                new_pos = cv2.morphologyEx((resize_gly*255).astype(np.uint8), cv2.MORPH_CLOSE, kernel=np.ones((resize_gly.shape[0]//10, resize_gly.shape[1]//10), dtype=np.uint8), iterations=1)
                new_pos = new_pos[..., np.newaxis] if len(new_pos.shape) == 2 else new_pos
                contours, _ = cv2.findContours(new_pos[..., 0:1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) != 1:
                    str_warning = f'Fail to revise position {i} to bounding rect, remain position unchanged...'
                else:
                    rect = cv2.minAreaRect(contours[0])
                    poly = np.int0(cv2.boxPoints(rect))
                    pre_pos[i] = cv2.drawContours(new_pos, [poly], -1, 255, -1) / 255.
                    gly_pos_img = cv2.drawContours(glyphs*255, [poly*gly_scale], 0, (255, 255, 255), 1)
                    
            gly_pos_imgs += [gly_pos_img]  # for show
        else:
            glyphs = np.zeros((h*gly_scale, w*gly_scale, 3))
            gly_line = np.zeros((80, 512, 1))
            gly_pos_imgs += [np.zeros((h*gly_scale, w*gly_scale, 1))]  # for show
            font_hint_line = np.zeros((h, w, 3))
        pos = pre_pos[i][..., 0:1]
        info['glyphs'] += [arr2tensor(glyphs, img_count, model.load_device, model.dtype)]
        info['gly_line'] += [arr2tensor(gly_line, img_count, model.load_device, model.dtype)]
        info['positions'] += [arr2tensor(pos, img_count, model.load_device, model.dtype)]
        info['colors'][i] = arr2tensor(info['colors'][i], img_count, model.load_device, model.dtype)/255.
        font_hint += [font_hint_line]
    font_hint_mimic_imgs = [font_hint_mimic_imgs] * img_count
    model.model.embedding_manager.font_hint_mimic_imgs = font_hint_mimic_imgs
    # get masked_x
    masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0 - np_hint*10).clip(-1, 1)
    masked_img = np.transpose(masked_img, (2, 0, 1))
    masked_img = torch.from_numpy(masked_img.copy()).float().cuda(0)
    
    # if self.use_fp16:
    #     masked_img = masked_img.half()
    masked_img = masked_img.to(model.dtype)
        
    encoder_posterior = model.model.encode_first_stage(masked_img[None, ...])
    masked_x = model.model.get_first_stage_encoding(encoder_posterior).detach()
    
    # if self.use_fp16:
    #     masked_x = masked_x.half()
    masked_x = masked_x.to(model.dtype)
        
    info['masked_x'] = torch.cat([masked_x for _ in range(img_count)], dim=0)

    hint = arr2tensor(np_hint, img_count, model.load_device, model.dtype)

    font_hint_fg = np.sum(font_hint, axis=0).clip(0, 1)[..., 0:1]*255
    if fonts.font_hollow and font_hint_fg.mean() > 0:
        img = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg_noise.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (font_hint_fg.shape[1], font_hint_fg.shape[0]))
        img[img < 230] = 0
        font_hint_bg = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel1 = np.ones((2, 2), dtype=np.uint8)
        kernel2 = np.ones((3, 3), dtype=np.uint8)
        dilate_img1 = cv2.dilate(font_hint_fg[..., 0].astype(np.uint8), kernel1, iterations=1)
        dilate_img2 = cv2.dilate(font_hint_fg[..., 0].astype(np.uint8), kernel2, iterations=1)
        dilate_text = dilate_img2 - dilate_img1
        result = (font_hint_fg[..., 0]-font_hint_bg + dilate_text).clip(0, 255)
        font_hint_bg[font_hint_fg[..., 0] > 0] = 0
        result = (result + font_hint_bg).clip(0, 255)
        font_hint_bg = result[..., None]
    else:
        font_hint_bg = font_hint_fg
        
    text_prompt = "#"
    for i in texts:
        text_prompt+=" *"

    info['font_hint'] = arr2tensor((font_hint_bg/255), img_count, model.load_device, model.dtype)
    cond = model.model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[[prompt + ', ' + a_prompt] * img_count, [text_prompt.replace("# ", "")] * img_count]], text_info=info))
    un_cond = model.model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[[n_prompt] * img_count, [""] * img_count]], text_info=info))
    
    
    font_img = []
    for f_i in gly_pos_imgs:
        f_i = torch.from_numpy(f_i).permute(2,0,1)
        font_img.append(f_i)
    font_img = torch.cat(font_img, dim=0)
    
    gly_line_img = []
    glyphs_img = []
    for g_la, g_lb in zip(gly_line_list, glyphs_list):
        g_la, g_lb = torch.from_numpy(g_la).permute(2,0,1), torch.from_numpy(g_lb).permute(2,0,1)
        gly_line_img.append(g_la)
        glyphs_img.append(g_lb)
    gly_line_img = torch.cat(gly_line_img, dim=0)
    glyphs_img = torch.cat(glyphs_img, dim=0)
    
    glyph_img = torch.from_numpy(np.sum(np.stack(gly_pos_imgs), axis=0).clip(0, 255).astype(np.uint8)).permute(2,0,1)
    
    font_hint_img = torch.from_numpy(np.repeat(font_hint_bg.astype(np.uint8), 3, axis=2)).permute(2,0,1)
        
    # return cond, un_cond, masked_x, font_img, gly_line_img, glyphs_img, masked_img.clone().cpu().float(), h, w, font_hint_img, glyph_img
    return cond, un_cond, h, w, font_hint_img, glyph_img