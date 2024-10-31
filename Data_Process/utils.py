import time
import torch
import time
import gc
from comfy.model_management import soft_empty_cache

def t5_translate_en_ru_zh(Target_Language, prompt, t5_translate_en_ru_zh_path, device, cpu_offload= False, loaded_t5_translate_en_ru_zh_tensor=None, loaded_t5_translate_en_ru_zh_model_name=None, keep_model_device=False):
    """_summary_

    Args:
        Target_Language (_type_): zh, en, ru
        prompt (_type_): _description_
        t5_translate_en_ru_zh_base_200_path (_type_): _description_
        device (_type_): _description_
        cpu_offload (bool, optional): _description_. Defaults to False.
        loaded_model_tensor (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: result, loaded_t5_translate_en_ru_zh_tensor, loaded_t5_translate_en_ru_zh_model_name
    """
    # prefix = 'translate to en: '
    sttime = time.time()
    from transformers import T5ForConditionalGeneration
    from transformers import T5Tokenizer
    
    if loaded_t5_translate_en_ru_zh_model_name != t5_translate_en_ru_zh_path:
        del loaded_t5_translate_en_ru_zh_tensor
        soft_empty_cache()
        t5_translate_en_ru_zh_tensor = None
        t5_translate_en_ru_zh_model_name = None
        loaded_t5_translate_en_ru_zh_tensor = None
    if loaded_t5_translate_en_ru_zh_tensor == None:
        t5_translate = T5ForConditionalGeneration.from_pretrained(t5_translate_en_ru_zh_path).to(device)
    else:
        t5_translate = loaded_t5_translate_en_ru_zh_tensor.to(device)
        
    tokenizer = T5Tokenizer.from_pretrained(t5_translate_en_ru_zh_path)
    if Target_Language == 'zh':
        prefix = 'translate to zh: '
    elif Target_Language == 'en':
        prefix = 'translate to en: '
    else:
        prefix = 'translate to ru: '
    src_text = prefix + prompt
    input_ids = tokenizer(src_text, return_tensors="pt")
    input_ids = input_ids.to(device)
    generated_tokens = t5_translate.generate(**input_ids).to(device, torch.float32)
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    if cpu_offload:
        if keep_model_device:
            t5_translate.to('cpu')
        else:
            t5_translate.to(device)
        t5_translate_en_ru_zh_tensor = t5_translate
        t5_translate_en_ru_zh_model_name = t5_translate_en_ru_zh_path
    else:
        del t5_translate
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t5_translate_en_ru_zh_tensor = None
        t5_translate_en_ru_zh_model_name = None
    endtime = time.time()
    print("\033[93mTime for translating(翻译耗时): ", endtime - sttime, "\033[0m")
    return result, t5_translate_en_ru_zh_tensor, t5_translate_en_ru_zh_model_name

def nlp_csanmt_translation_zh2en(device, prompt, nlp_csanmt_translation_zh2en_path):
    sttime = time.time()
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    pipeline_ins = pipeline(task=Tasks.translation, model=nlp_csanmt_translation_zh2en_path, device=device)
    outputs = pipeline_ins(input=prompt)
    endtime = time.time()
    from tensorflow.python.client import device_lib
    print("\033[93m翻译耗时：", endtime - sttime, "\n翻译使用的设备：\n", device_lib.list_local_devices(), "\033[0m")
    del pipeline_ins
    soft_empty_cache()
    return outputs