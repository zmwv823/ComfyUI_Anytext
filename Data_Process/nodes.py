import os
import folder_paths
import time
import torch
from ..UL_common.common import get_dtype_by_name
from comfy.model_management import get_torch_device, text_encoder_offload_device, soft_empty_cache
from comfy.utils import ProgressBar

current_directory = os.path.dirname(os.path.abspath(__file__))
    
class UL_TranslatorLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["utrobinmv/t5_translate_en_ru_zh_small_1024", "iic/nlp_csanmt_translation_zh2en", "iic/nlp_csanmt_translation_en2zh", "iic/nlp_csanmt_translation_en2zh_base", "botisan-ai/mt5-translate-zh-yue", "facebook/nllb-200-distilled-600M", "facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-3.3B"], {"default": "utrobinmv/t5_translate_en_ru_zh_small_1024", "tooltip": ""}),
                "weight_dtype": (["auto", "fp16", "bf16", "fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto", "tooltip": "For t5、nllb."}),
                "Auto_Download_Path": ("BOOLEAN", {"default": True, "label_on": "models_local本地", "label_off": ".cache缓存", "tooltip": "Download to `ComfyUI\models\prompt_generator` or huggingface cache dir.\n下载模型到`ComfyUI\models\prompt_generator`或者缓存到huggingface缓存路径。"}),
            }
        }

    RETURN_TYPES = ("TRANSLATE_MODEL", "STRING", )
    RETURN_NAMES = ("model", "model_dir", )
    CATEGORY = "UL Group/Data Process"
    FUNCTION = "main"
    TITLE = "Translator Loader"
    DESCRIPTION = ""
    
    def main(self, model_name, weight_dtype, Auto_Download_Path):
        dtype = get_dtype_by_name(weight_dtype)
        prompt_generator_dir = os.path.join(folder_paths.models_dir, "prompt_generator")
        if 'nlp_csanmt_translation' in model_name.lower():
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            
            nlp_device = 'cuda' if 'cuda' in get_torch_device().type else "cpu"
            
            model_dir = os.path.join(prompt_generator_dir, "modelscope--damo--nlp_csanmt_translation_zh2en") if "zh2en" in model_name.lower() else os.path.join(prompt_generator_dir, "modelscope--damo--nlp_csanmt_translation_en2zh_base") if "en2zh_base" in model_name.lower() else os.path.join(prompt_generator_dir, "modelscope--damo--nlp_csanmt_translation_en2zh")
            
            if not os.path.exists(os.path.join(model_dir, "tf_ckpts", "ckpt-0.data-00000-of-00001")):
                if Auto_Download_Path:
                    from modelscope.hub.snapshot_download import snapshot_download
                    snapshot_download(
                        model_id=model_name,
                        local_dir=model_dir,
                    )
                else:
                    model_dir = model_name
            
            model = pipeline(task=Tasks.translation, model=model_dir, device=nlp_device)
            tokenizer = "None"
            model_type = "nlp"
        elif model_name == "botisan-ai/mt5-translate-zh-yue":
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            model_dir = os.path.join(prompt_generator_dir, "models--botisan-ai--mt5-translate-zh-yue")
            
            if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
                if Auto_Download_Path:
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=model_dir,
                        ignore_patterns=["*.bin"]
                    )
                else:
                    model_dir = model_name
                    
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            model_type = "mt5"
        elif "nllb-200" in model_name.lower():
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            
            model_dir = os.path.join(prompt_generator_dir, f'models--{model_name.replace("/", "--")}')
            if not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
                if Auto_Download_Path:
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=model_dir,
                    )
                else:
                    model_dir = model_name
                    
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(dtype)
            model_type = "nllb"
        elif "t5_translate_en_ru_zh" in model_name.lower():
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            model_dir = os.path.join(prompt_generator_dir, f'models--{model_name.replace("/", "--")}')
            if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
                if Auto_Download_Path:
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=model_dir,
                    )
                else:
                    model_dir = model_name
            
            tokenizer = T5Tokenizer.from_pretrained(model_dir)
            model = T5ForConditionalGeneration.from_pretrained(model_dir).to(dtype)
            model_type = "t5"
            
        model = {
            "model": model,
            "tokenizer": tokenizer,
            "model_type": model_type,
            "dtype": dtype,
        }
            
        return (model, model_dir, )
    
class UL_Translator:
    @classmethod
    def INPUT_TYPES(cls):
        from .Translate_Scripts.translate_backbone import language_list
        return {
            "required": {
                "model": ("TRANSLATE_MODEL", ),
                "string": ("STRING", {'default': '', "multiline": True, "dynamicPrompts": True}),
                "src_language": (language_list, {"default": "", "tooltip": "For nllb."}),
                "tgt_language": (language_list, {"default": "", "tooltip": "For nllb、t5."}),
                "max_new_tokens": ("INT", {"default": 512, "min": 0, "max": 0xffffffffffffffff, "tooltip": "For mt5、nllb."}),
                "use_fast": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "For nllb."}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Warning: do not delete model unless this node no longer needed, it will try release device_memory and ram. if checked and want to continue node generation, use ComfyUI-Manager `Free model and node cache` to reset node state or change parameter in Loader node to activate.\n注意：仅在这个节点不再需要时删除模型，将尽量尝试释放系统内存和设备专用内存。如果删除后想继续使用此节点，使用ComfyUI-Manager插件的`Free model and node cache`重置节点状态或者更换模型加载节点的参数来激活。"}),
                "keep_model_device": ("BOOLEAN", {"default": True, "label_on": "comfy", "label_off": "device", "tooltip": "Keep model in comfy_auto_unet_offload_device (HIGH_VRAM: device, Others: cpu) or device_memory after generation.\n生图完成后，模型转移到comfy自动选择设备(HIGH_VRAM: device, 其他: cpu)或者保留在设备专用内存上。"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    CATEGORY = "UL Group/Data Process"
    FUNCTION = "main"
    TITLE = "Translator"
    DESCRIPTION = ""
    
    def main(self, model, string, src_language, tgt_language, max_new_tokens, keep_model_loaded, keep_model_device, use_fast):
        from .Translate_Scripts.translate_backbone import nlp_tranlate, mt5_translate, nllb_translate, t5_language_map, t5_translate
        sttime = time.time()
        input_text = [string]
        output_text = ""
        if "\n" in string:
            input_text = []
            for txt in string.split("\n"):
                input_text.append(txt)
        pbar = ProgressBar(len(input_text))
                
        for input in input_text:
            if model['model_type'] == "nlp":
                result = nlp_tranlate(model['model'], input)
                output_text+=f'{result}\n'
                pbar.update(1)
            elif model['model_type'] == "mt5":
                result = mt5_translate(model['model'], model['tokenizer'], input, max_new_tokens)
                output_text+=f'{result}\n'
                pbar.update(1)
            elif model['model_type'] == "nllb":
                if src_language != tgt_language:
                    result = nllb_translate(model['model'], model['tokenizer'], input, src_language, tgt_language, use_fast, max_new_tokens)
                    output_text+=f'{result}\n'
                    pbar.update(1)
                else:
                    output_text = string
            elif model['model_type'] == "t5":
                if tgt_language in t5_language_map.keys():
                    result = t5_translate(model['model'], model['tokenizer'], input, t5_language_map[tgt_language])
                    output_text+=f'{result}\n'
                    pbar.update(1)
                else:
                    output_text = string
                
        endtime = time.time()
        print("\n\033[93m翻译耗时：", endtime - sttime, "\033[0m")
        if keep_model_loaded and keep_model_device and torch.cuda.is_available():
            if model['model_type'] != "nlp":
                model['model'].to(text_encoder_offload_device())
            soft_empty_cache(True)
        elif not keep_model_loaded:
            del model['model'], model['tokenizer']
            soft_empty_cache(True)
            
        return (output_text, )

# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "UL_TranslatorLoader": UL_TranslatorLoader,
    "UL_Translator": UL_Translator,
}

target_language_list = ["source", "zh",  "yue", "en", "ru", "ja", "ko", "es", "fr", "it", "uk", "hi", "ar", "af", "am", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "et", "eu", "fa", "fi", "fo", "gl", "gu", "ha", "haw", "he", "hr", "ht", "hu", "hy", "id", "is", "jw", "ka", "kk", "km", "kn", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "ur", "uz", "vi", "yi", "yo"]