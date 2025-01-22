from comfy.model_management import get_torch_device

t5_language_map = {
    "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh",
    "English": "en",
    "Russian": "ru",
}

language_list = ["Chinese (Simplified)", "English", "Yue Chinese", "Chinese (Traditional)", "Standard Tibetan", "Japanese", "Korean", "Spanish", "Russian", "Modern Standard Arabic", "French", "German", "Italian", "Portuguese", "Polish", "Turkish", "Dutch", "Czech", "Hungarian", "Hindi", "Halh Mongolian", "Uyghur"]

def nlp_tranlate(model, text):
    from tensorflow.python.client import device_lib
    result = model(input=text)['translation']
    print("\033[93m翻译使用的设备：\n", device_lib.list_local_devices(), ".\033[0m")
    return result

def mt5_translate(model, tokenizer, text, max_new_tokens):
    model.to(get_torch_device())
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(get_torch_device())  # Batch size 1
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result

def t5_translate(model, tokenizer, text, tgt_language):
    if tgt_language == 'zh':
        prefix = 'translate to zh: '
    elif tgt_language == 'en':
        prefix = 'translate to en: '
    else:
        prefix = 'translate to ru: '
    model.to(get_torch_device())
    src_text = prefix + text
    input_ids = tokenizer(src_text, return_tensors="pt")
    generated_tokens = model.generate(**input_ids.to(get_torch_device()))
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return result

def nllb_translate(model, tokenizer, text, src_language, tgt_language, use_fast, max_new_tokens):
    from .nllb_200__flores200_codes import flores_codes
    from transformers import pipeline
    model.to(get_torch_device())
    src_language, tgt_language = flores_codes[src_language], flores_codes[tgt_language]
    pipe = pipeline('translation', model, tokenizer=tokenizer, src_lang=src_language, tgt_lang=tgt_language, device_map=get_torch_device(), torch_dtype=model.dtype, use_fast=use_fast)
    result = pipe(text, max_length=max_new_tokens)[0]['translation_text']
    return result