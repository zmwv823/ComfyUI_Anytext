'''
Copyright (c) Alibaba, Inc. and its affiliates.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from AnyText2ControlDiffusion.ldm.modules.diffusionmodules.util import conv_nd, linear, zero_module
import numpy as np
from AnyText2ControlDiffusion.cldm.recognizer import crop_image, TextRecognizer, create_predictor
import math
from easydict import EasyDict as edict
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"
    return tokens[0, 1]


def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"
    token = token[0, 1]
    return token


def get_clip_vision_emb(encoder, processor, img):
    _img = img.repeat(1, 3, 1, 1)*255
    inputs = processor(images=_img, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].to(img.device)
    outputs = encoder(**inputs)
    emb = outputs.image_embeds
    return emb


def get_recog_emb(encoder, img_list):
    repeat = 3 if img_list[0].shape[1] == 1 else 1
    _img_list = [(img.repeat(1, repeat, 1, 1)*255)[0] for img in img_list]
    encoder.predictor.eval()
    _, preds_neck = encoder.pred_imglist(_img_list, show_debug=False)
    return preds_neck


def get_style_emb(encoder, img_list):
    repeat = 3 if img_list[0].shape[1] == 1 else 1
    _img_list = [(img.repeat(1, repeat, 1, 1)*255)[0] for img in img_list]
    # encoder.predictor.train()
    _, preds_neck = encoder.pred_imglist(_img_list, show_debug=False)
    return preds_neck


def pad_H(x):
    _, _, H, W = x.shape
    p_top = (W - H) // 2
    p_bot = W - H - p_top
    return F.pad(x, (0, 0, p_top, p_bot))


# img: CHW, result: CHW tensor 0-255
def resize_img(img, imgH, imgW):

    c, h, w = img.shape
    if h > w * 1.2:
        img = torch.transpose(img, 1, 2).flip(dims=[1])
        h, w = img.shape[1:]

    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = torch.nn.functional.interpolate(
        img.unsqueeze(0),
        size=(imgH, resized_w),
        mode='bilinear',
        align_corners=True,
    )
    padding_im = torch.zeros((c, imgH, imgW), dtype=torch.float32).to(img.device)
    padding_im[:, :, 0:resized_w] = resized_image[0]
    return padding_im


class EncodeNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodeNet, self).__init__()
        chan = 16
        n_layer = 4  # downsample

        self.conv1 = conv_nd(2, in_channels, chan, 3, padding=1)
        self.conv_list = nn.ModuleList([])
        _c = chan
        for i in range(n_layer):
            self.conv_list.append(conv_nd(2, _c, _c*2, 3, padding=1, stride=2))
            _c *= 2
        self.conv2 = conv_nd(2, _c, out_channels, 3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        for layer in self.conv_list:
            x = self.act(layer(x))
        x = self.act(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            valid=True,
            glyph_channels=20,
            position_channels=1,
            style_channels=1,
            placeholder_string='*',
            add_pos=False,
            emb_type='ocr',
            big_linear=False,
            add_style_conv=False,
            add_style_ocr=False,
            enable_flag=True,
            color_fourier_encode=False,
            style_ocr_trainable=True,
            add_color=False,
            color_big_linear=False,
            **kwargs
    ):
        super().__init__()
        if hasattr(embedder, 'tokenizer'):  # using Stable Diffusion's CLIP encoder
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            token_dim = 768
            if hasattr(embedder, 'vit'):
                assert emb_type == 'vit'
                self.get_vision_emb = partial(get_clip_vision_emb, embedder.vit, embedder.processor)
            self.get_recog_emb = None
        else:  # using LDM's BERT encoder
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            token_dim = 1280
        self.token_dim = token_dim
        self.emb_type = emb_type
        self.big_linear = big_linear

        self.add_pos = add_pos
        self.add_style_conv = add_style_conv
        self.add_style_ocr = add_style_ocr
        self.enable_flag = enable_flag
        self.color_fourier_encode = color_fourier_encode
        self.style_ocr_trainable = style_ocr_trainable
        assert not (self.add_style_conv and self.add_style_ocr)
        self.add_color = add_color
        self.color_big_linear = color_big_linear
        if add_pos:
            self.position_encoder = EncodeNet(position_channels, token_dim)
        if add_style_conv:
            self.style_encoder = EncodeNet(style_channels, token_dim)
        if add_style_ocr:
            if self.style_ocr_trainable:
                self.font_predictor = create_predictor()
                args = edict()
                args.rec_image_shape = "3, 48, 320"
                args.rec_batch_num = 6
                args.rec_char_dict_path = './ocr_recog/ppocr_keys_v1.txt'
                args.use_fp16 = False
                self.style_encoder = TextRecognizer(args, self.font_predictor)
                for param in self.font_predictor.parameters():
                    param.requires_grad = True
            else:
                self.style_encoder = None
            self.style_proj = nn.Sequential(
                                zero_module(linear(40*64, token_dim)),
                                nn.LayerNorm(token_dim)
                                )
            self.get_style_emb = None
        if add_color:
            if self.color_fourier_encode:
                self.rgb_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
                self.rgb_encoder = TimestepEmbedding(in_channels=256*3, time_embed_dim=token_dim)
            else:
                if self.color_big_linear:
                    self.color_proj = nn.Sequential(
                                zero_module(linear(3, 1280)),
                                nn.SiLU(),
                                zero_module(linear(1280, token_dim)),
                                nn.LayerNorm(token_dim)
                                )
                else:
                    self.color_proj = nn.Sequential(
                                        zero_module(linear(3, token_dim)),
                                        nn.LayerNorm(token_dim)
                                        )
        if emb_type == 'ocr':
            if big_linear:
                self.proj = nn.Sequential(
                                linear(40*64, 1280),
                                nn.SiLU(),
                                linear(1280, token_dim),
                                nn.LayerNorm(token_dim)
                                )
            else:
                self.proj = nn.Sequential(
                                zero_module(linear(40*64, token_dim)),
                                nn.LayerNorm(token_dim)
                                )
        if emb_type == 'conv':
            self.glyph_encoder = EncodeNet(glyph_channels, token_dim)

        self.placeholder_token = get_token_for_string(placeholder_string)
        self.font_hint_mimic_imgs = None
        self.start_idx = None

    def reset_start_idx(self):
        self.start_idx = [0] * len(self.text_embs_all)

    def encode_text(self, text_info):
        if self.get_recog_emb is None and self.emb_type == 'ocr':
            self.get_recog_emb = partial(get_recog_emb, self.recog)
        if self.add_style_ocr:
            if self.get_style_emb is None:
                if self.style_encoder is None:  # not trainable
                    self.style_encoder = self.recog
                self.get_style_emb = partial(get_style_emb, self.style_encoder)

        gline_list = []
        pos_list = []
        style_list = []
        color_list = []
        style_flag = []
        color_flag = []
        for i in range(len(text_info['n_lines'])):  # sample index in a batch
            n_lines = text_info['n_lines'][i]
            for j in range(n_lines):  # line
                gline_list += [text_info['gly_line'][j][i:i+1]]
                if self.add_pos:
                    pos_list += [text_info['positions'][j][i:i+1]]
                if self.add_style_conv:
                    np_pos = text_info['positions'][j][i].permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255  # hwc, numpy, 0-255
                    font_hint = (text_info['font_hint'][i]*255)  # 1hw, tensor, 0-255
                    style_line = crop_image(font_hint, np_pos)/255.  # 1hw, tensor, 0-1
                    style_line = resize_img(style_line, imgH=48, imgW=320)[None, ...]  # 11HW tensor 0-1  48x320
                    style_line = style_line.to(text_info['positions'][j][i].dtype)
                    style_list += [style_line]
                if self.add_style_ocr:
                    np_pos = text_info['positions'][j][i].permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255  # hwc, numpy, 0-255
                    font_hint = (text_info['font_hint'][i]*255)
                    if font_hint.shape[0] == 1:
                        font_hint = font_hint.repeat(3, 1, 1)  # chw, tensor, 0-255
                    if self.font_hint_mimic_imgs is not None and isinstance(self.font_hint_mimic_imgs, list):
                        if (i < len(self.font_hint_mimic_imgs) and
                            isinstance(self.font_hint_mimic_imgs[i], list) and
                            j < len(self.font_hint_mimic_imgs[i])):
                            mimic_img = self.font_hint_mimic_imgs[i][j]
                        else:
                            mimic_img = None
                    else:
                        mimic_img = None
                    if mimic_img is not None:  # chw, tensor, 0-255
                        style_line = (mimic_img.to(font_hint.device)/255.)[None, ...]  # 1chw, tensor, 0-1
                    else:
                        style_line = (crop_image(font_hint, np_pos)/255.)[None, ...]  # 1chw, tensor, 0-1
                    if style_line.mean() == 0:
                        style_flag += [0]
                    else:
                        style_flag += [1]
                    style_list += [style_line]
                if self.add_color:
                    _c = text_info['colors'][j][i:i+1]
                    if _c.mean() > 1 or _c.mean() < 0:
                        color_flag += [0]
                    else:
                        color_flag += [1]
                    color_list += [_c]

        if len(gline_list) > 0:
            if self.emb_type == 'ocr':
                recog_emb = self.get_recog_emb(gline_list)
                enc_glyph = self.proj(recog_emb.reshape(recog_emb.shape[0], -1))
            elif self.emb_type == 'vit':
                enc_glyph = self.get_vision_emb(pad_H(torch.cat(gline_list, dim=0)))
            elif self.emb_type == 'conv':
                enc_glyph = self.glyph_encoder(pad_H(torch.cat(gline_list, dim=0)))
            if self.add_pos:
                enc_pos = self.position_encoder(torch.cat(pos_list, dim=0))
                enc_glyph = enc_glyph+enc_pos
            if self.add_style_conv:
                enc_style = self.style_encoder(torch.cat(style_list, dim=0))
                enc_glyph = enc_glyph+enc_style
            if self.add_style_ocr:
                style_emb = self.get_style_emb(style_list)
                enc_style = self.style_proj(style_emb.reshape(style_emb.shape[0], -1))
                mask_style = torch.tensor(style_flag, dtype=torch.bool).to(enc_style.device)
                if self.enable_flag:
                    enc_style[~mask_style] = 0
                enc_glyph = enc_glyph+enc_style
            if self.add_color:
                rgb_batch = torch.cat(color_list, dim=0)*255.
                if self.color_fourier_encode:
                    enc_color = self.rgb_proj(rgb_batch.flatten())
                    dtype = next(self.rgb_encoder.parameters()).dtype
                    enc_color = enc_color.reshape((rgb_batch.shape[0], -1)).to(rgb_batch.device).to(dtype)
                    enc_color = self.rgb_encoder(enc_color)
                else:
                    enc_color = self.color_proj(rgb_batch)
                mask_color = torch.tensor(color_flag, dtype=torch.bool).to(enc_style.device)
                if self.enable_flag:
                    enc_color[~mask_color] = 0
                enc_glyph = enc_glyph+enc_color

        self.text_embs_all = []
        n_idx = 0
        for i in range(len(text_info['n_lines'])):  # sample index in a batch
            n_lines = text_info['n_lines'][i]
            text_embs = []
            for j in range(n_lines):  # line
                text_embs += [enc_glyph[n_idx:n_idx+1]]
                n_idx += 1
            self.text_embs_all += [text_embs]
        self.reset_start_idx()

    def forward(
            self,
            tokenized_text,
            embedded_text,
    ):
        b, device = tokenized_text.shape[0], tokenized_text.device
        for i in range(b):
            idx = tokenized_text[i] == self.placeholder_token.to(device)
            if sum(idx) > 0:
                if i >= len(self.text_embs_all):
                    print('truncation for log images...')
                    break
                text_emb = self.text_embs_all[i][self.start_idx[i]:sum(idx)+self.start_idx[i]]
                text_emb = torch.cat(text_emb, dim=0)
                embedded_text[i][idx] = text_emb
                self.start_idx[i] += sum(idx)
        return embedded_text

    def embedding_parameters(self):
        return self.parameters()
