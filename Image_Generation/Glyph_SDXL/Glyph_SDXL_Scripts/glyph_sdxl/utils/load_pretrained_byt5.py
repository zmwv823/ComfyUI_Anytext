import json

from transformers import AutoTokenizer, T5ForConditionalGeneration
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def add_special_token(tokenizer, text_encoder, add_color, add_font, color_ann_path, font_ann_path, multilingual=False):
    with open(font_ann_path, 'r') as f:
        idx_font_dict = json.load(f)
    with open(color_ann_path, 'r') as f:
        idx_color_dict = json.load(f)

    if multilingual:
        font_token = []
        for font_code in idx_font_dict:
            prefix = font_code[:2]
            font_token.append(f'<{prefix}-font-{idx_font_dict[font_code]}>')
    else:
        font_token = [f'<font-{i}>' for i in range(len(idx_font_dict))]
    color_token = [f'<color-{i}>' for i in range(len(idx_color_dict))]
    additional_special_tokens = []
    if add_color:
        additional_special_tokens += color_token
    if add_font:
        additional_special_tokens += font_token
    tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
    text_encoder.resize_token_embeddings(len(tokenizer))

def load_byt5_and_byt5_tokenizer(
    byt5_name='google/byt5-small', 
    special_token=False, 
    color_special_token=False,
    font_special_token=False,
    color_ann_path='assets/color_idx.json',
    font_ann_path='assets/font_idx_512.json',
    huggingface_cache_dir=None,
    multilingual=False,
):
    byt5_tokenizer = AutoTokenizer.from_pretrained(
        byt5_name, cache_dir=huggingface_cache_dir,
    )
    byt5_text_encoder = T5ForConditionalGeneration.from_pretrained(
        byt5_name, cache_dir=huggingface_cache_dir,
    ).get_encoder()

    if special_token:
        add_special_token(
            byt5_tokenizer, 
            byt5_text_encoder, 
            add_color=color_special_token, 
            add_font=font_special_token, 
            color_ann_path=color_ann_path,
            font_ann_path=font_ann_path,
            multilingual=multilingual,
        )

    logger.info(f'Loaded original byt5 weight')
    
    return byt5_text_encoder, byt5_tokenizer