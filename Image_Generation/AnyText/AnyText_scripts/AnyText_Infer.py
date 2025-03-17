import cv2
import numpy as np
import torch
from .AnyText_pipeline_util import resize_image
from ....UL_common.common import tensor2numpy_cv2, numpy_cv2tensor

def encode(
    model, font, mask, prompt, texts, latent, mode, sort_radio, a_prompt, n_prompt, revise_pos, random_mask, Random_Gen_Mask_path, image, **extra_kwargs
):
    
    from .AnyText_pipeline import separate_pos_imgs, find_polygon, draw_glyph, draw_glyph2
    from .AnyText_pipeline_util import check_channels
    
    max_chars = 50
    batch_size, height, width = latent["samples"].shape[0], latent["samples"].shape[2] * 8, latent["samples"].shape[3] * 8 # B, C, H, W
    
    dtype = model.dtype #model.model.diffusion_model.dtype
                
    if not random_mask:
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
    
    if random_mask:
        mask = generate_rectangles(width, height, n_lines, max_trys=500, Random_Gen_Mask_path=Random_Gen_Mask_path)
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
    gly_line_list = []
    glyphs_list = []
    for i in range(len(texts)):
        text = texts[i]
        if len(text) > max_chars:
            print(f'"{text}" length > max_chars: {max_chars}, will be cut off...')
            text = text[:max_chars]
        gly_scale = 2
        if pre_pos[i].mean() != 0:
            gly_line = draw_glyph(font, text)
            glyphs, _ = draw_glyph2(font, text, poly_list[i], scale=gly_scale, width=w, height=h, add_space=False)
            gly_line_list+=[gly_line]
            glyphs_list+=[glyphs]
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
        info['glyphs'] += [arr2tensor(glyphs, batch_size, dtype, model.load_device)]
        info['gly_line'] += [arr2tensor(gly_line, batch_size, dtype, model.load_device)]
        info['positions'] += [arr2tensor(pos, batch_size, dtype, model.load_device)]
        
    # get masked_x
    masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0)*(1-np_hint)
    masked_img = np.transpose(masked_img, (2, 0, 1))
    masked_img = torch.from_numpy(masked_img.copy()).float()
    masked_img = masked_img.to(model.load_device, dtype)
    
    encoder_posterior = model.model.encode_first_stage(masked_img[None, ...])
    masked_x = (model.model.get_first_stage_encoding(encoder_posterior).detach()).to(dtype)
    
    info['masked_x'] = torch.cat([masked_x for _ in range(batch_size)], dim=0)
    hint = arr2tensor(np_hint, batch_size, dtype, model.load_device)
    
    cond = model.model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[anytext_prompt + ' , ' + a_prompt] * batch_size], text_info=info))
    un_cond = model.model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[n_prompt] * batch_size], text_info=info))
    
    # font_img = []
    # for f_i in gly_pos_imgs:
    #     f_i = torch.from_numpy(f_i).permute(2,0,1)
    #     font_img.append(f_i)
    # font_img = torch.cat(font_img, dim=0)
    
    gly_line_img = []
    glyphs_img = []
    # for g_la, g_lb in zip(gly_line_list, glyphs_list):
    #     g_la, g_lb = torch.from_numpy(g_la).permute(2,0,1), torch.from_numpy(g_lb).permute(2,0,1)
    #     gly_line_img.append(g_la)
    #     glyphs_img.append(g_lb)
    # gly_line_img = torch.cat(gly_line_img, dim=0)
    # glyphs_img = torch.cat(glyphs_img, dim=0)
    
    # return cond, un_cond, masked_x, font_img, gly_line_img, glyphs_img, masked_img.clone().cpu().float(), h, w, font_img, mask
    return cond, un_cond, h, w, glyphs_img, gly_line_img

def arr2tensor(arr, bs, dtype, device):
    arr = np.transpose(arr, (2, 0, 1))
    _arr = torch.from_numpy(arr.copy()).float().to(device)
    _arr = (torch.stack([_arr for _ in range(bs)], dim=0)).to(dtype)
    return _arr

def generate_rectangles(w, h, n, max_trys=200, Random_Gen_Mask_path: str=""):
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

def check_overlap_polygon(rect_pts1, rect_pts2):
    poly1 = cv2.convexHull(rect_pts1)
    poly2 = cv2.convexHull(rect_pts2)
    rect1 = cv2.boundingRect(poly1)
    rect2 = cv2.boundingRect(poly2)
    if rect1[0] + rect1[2] >= rect2[0] and rect2[0] + rect2[2] >= rect1[0] and rect1[1] + rect1[3] >= rect2[1] and rect2[1] + rect2[3] >= rect1[1]:
        return True
    return False