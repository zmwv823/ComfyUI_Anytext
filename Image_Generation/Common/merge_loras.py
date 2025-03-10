import folder_paths
import time
from comfy.utils import load_torch_file, ProgressBar
import torch
from tqdm import tqdm
from copy import deepcopy

class U_LoRAS:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("AnyText_Model", ),
                "lora_name": (loras,),
                "weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "", "advanced": False}),
                
                "lora_name1": (loras,),
                "weight1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch1": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "", "advanced": False}),
                
                "lora_name2": (loras,),
                "weight2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch2": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "", "advanced": False}),
                
                "lora_name3": (loras,),
                "weight3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch3": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "", "advanced": False}),
                
                "lora_name4": (loras,),
                "weight4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch4": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "", "advanced": False}),
                
                # "lora_name_5": (loras,),
                # "weight_5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                # "switch_5": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "", "advanced": False}),
                
                # "lora_name_6": (loras,),
                # "weight_6": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                # "switch_6": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "", "advanced": False}),
                
                # "lora_name_7": (loras,),
                # "weight_7": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                # "switch_7": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "", "advanced": False}),
        },
    }

    RETURN_TYPES = ("AnyText_Model", "STRING", )
    RETURN_NAMES = ("model", "path", )
    FUNCTION = "main"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "Loras"
    DESCRIPTION = "Don't bypass or mute if lora not needed after lora applied, turn off the switch to reset model state_dict, or the model will be poluted.\n如果已经应用lora，然后不再需要lora，不要bypass或者mute，使用开关来重置模型，否则模型会被污染。"
    
    def __init__(self):
        self.paths = None
        self.model_load_kwargs = None
        self.model_clone = None
        self.clip_clone = None

    def main(self, model, **kwargs):
        lora_paths = []
        lora_ratios = []
        switchs = []
        for k in list(kwargs.values()):
            if isinstance(k, bool):
                switchs.append(k)
            if isinstance(k, str):
                k = folder_paths.get_full_path_or_raise("loras", k)
                lora_paths.append(k)
            if isinstance(k, float):
                lora_ratios.append(k)
                
        paths = []
        ratios = []
        for i, j, k in zip(lora_paths, lora_ratios, switchs):
            if k and i not in paths:
                paths.append(i)
                ratios.append(j)
        
        if (self.model_load_kwargs != model.load_kwargs or model.model_clone == None) and len(paths)>0:
            self.model_load_kwargs = model.load_kwargs
            self.model_clone = self.clip_clone = model.model_clone = model.clip_clone = None
            self.model_clone = deepcopy(model.model.model.diffusion_model)
            self.clip_clone = deepcopy(model.model.cond_stage_model.transformer)
            
        if model.model_clone != None:
            model.model.model.diffusion_model = self.model_clone
            model.model.cond_stage_model.transformer = self.clip_clone
            
        if len(paths)==0:
            self.model_clone = self.clip_clone = model.model_clone = model.clip_clone = None
            
        model.model_clone, model.clip_clone = self.model_clone, self.clip_clone
            
        if self.paths != paths and len(paths)>0:
            self.paths = str(paths)+model.load_kwargs
            model.custom_load([model.model.cond_stage_model, model.model.model.diffusion_model], force_load=True)
            merge_loras(model, paths, ratios, model.load_device)
            model.custom_offload([model.model], do_clean=True, force_offload=True)

        return (model, str(paths), )
    
NODE_CLASS_MAPPINGS = {
    "U_LoRAS": U_LoRAS,
}

'''
Borrowed and modified from sd-scripts, publicly available at
https://github.com/kohya-ss/sd-scripts/blob/main/networks/merge_lora.py
'''
def merge_loras(model, lora_paths, lora_ratios, device):
    tic = time.time()
    from .Scripts.lora_util import get_diffusers_unet, convert_unet_state_dict_to_sd
    
    assert lora_paths is not None and len(lora_paths) == len(lora_ratios)
    
    if "AnyText" in model.model_type:
        unet = get_diffusers_unet(unet=get_diffusers_unet(), state_dict=model.model.state_dict()).to(device)#.cpu()
        text_encoder = model.model.cond_stage_model.transformer.to(device)#.cpu()

    # create module map
    name_to_module = {}
    for i, root_module in enumerate([text_encoder, unet]):
        if i == 0:
            prefix = "lora_te"
            target_replace_modules = ["CLIPAttention", "CLIPMLP"]
        else:
            prefix = "lora_unet"
            target_replace_modules = ["Transformer2DModel", "Attention"] + ["ResnetBlock2D", "Downsample2D", "Upsample2D"]

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        name_to_module[lora_name] = child_module
                        
    pbar = ProgressBar(len(lora_paths))
    with tqdm(total=len(lora_paths), desc="Merging loras...") as pbar_T:
        for lora, ratio in zip(lora_paths, lora_ratios):
            print(f"loading lora: {lora}")
            lora_sd = load_torch_file(lora, safe_load=True, device=device)

            print("merging...")
            for key in lora_sd.keys():
                if "lora_down" in key:
                    up_key = key.replace("lora_down", "lora_up")
                    alpha_key = key[: key.index("lora_down")] + "alpha"

                    # find original module for this lora
                    module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
                    if module_name not in name_to_module:
                        # print(f"no module found for LoRA weight: {key}")
                        continue
                    module = name_to_module[module_name]
                    # print(f"apply {key} to {module}")

                    down_weight = lora_sd[key]
                    up_weight = lora_sd[up_key]

                    dim = down_weight.size()[0]
                    alpha = lora_sd.get(alpha_key, dim)
                    scale = alpha / dim

                    # W <- W + U * D
                    weight = module.weight
                    dtype = weight.dtype
                    if len(weight.size()) == 2:
                        # linear
                        # weight = weight + ratio * (up_weight @ down_weight) * scale
                        weight = weight.float() + ratio * (up_weight.float() @ down_weight.float()) * scale
                        weight = weight.to(dtype)
                    elif down_weight.size()[2:4] == (1, 1):
                        # conv2d 1x1
                        weight = (weight.float() + ratio * (up_weight.float().squeeze(3).squeeze(2) @ down_weight.float().squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * scale.float())
                        weight = weight.to(dtype)
                    else:
                        # conv2d 3x3
                        conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                        # print(conved.size(), weight.size(), module.stride, module.padding)
                        weight = weight + ratio * conved * scale
                    module.weight = torch.nn.Parameter(weight)
            del lora_sd
            pbar.update(1)
            pbar_T.update()
            
    sd_from_diffuser = convert_unet_state_dict_to_sd(unet.state_dict())
    del unet
    
    # load new state_dict
    if "AnyText" in model.model_type:
        info_te = model.model.cond_stage_model.transformer.load_state_dict(text_encoder.state_dict())
        try:
            info_unet = model.model.model.diffusion_model.load_state_dict(sd_from_diffuser)
        except Exception as e:
            for k in missing_keys:
                sd_from_diffuser[k] = model.model.model.diffusion_model.state_dict()[k]
            info_unet = model.model.model.diffusion_model.load_state_dict(sd_from_diffuser)
        
    del sd_from_diffuser
        
    print(f'Merge lora model(s) done! text_encoder:{info_te}, unet:{info_unet}, cost time={(time.time()-tic)*1000.:.2f}ms')
    
missing_keys = ["middle_block.1.transformer_blocks.0.attn1x.to_q.weight", "middle_block.1.transformer_blocks.0.attn1x.to_k.weight", "middle_block.1.transformer_blocks.0.attn1x.to_v.weight", "middle_block.1.transformer_blocks.0.attn1x.to_out.0.weight", "middle_block.1.transformer_blocks.0.attn1x.to_out.0.bias", "middle_block.1.transformer_blocks.0.attn2x.to_q.weight", "middle_block.1.transformer_blocks.0.attn2x.to_k.weight", "middle_block.1.transformer_blocks.0.attn2x.to_v.weight", "middle_block.1.transformer_blocks.0.attn2x.to_out.0.weight", "middle_block.1.transformer_blocks.0.attn2x.to_out.0.bias", "output_blocks.3.1.transformer_blocks.0.attn1x.to_q.weight", "output_blocks.3.1.transformer_blocks.0.attn1x.to_k.weight", "output_blocks.3.1.transformer_blocks.0.attn1x.to_v.weight", "output_blocks.3.1.transformer_blocks.0.attn1x.to_out.0.weight", "output_blocks.3.1.transformer_blocks.0.attn1x.to_out.0.bias", "output_blocks.3.1.transformer_blocks.0.attn2x.to_q.weight", "output_blocks.3.1.transformer_blocks.0.attn2x.to_k.weight", "output_blocks.3.1.transformer_blocks.0.attn2x.to_v.weight", "output_blocks.3.1.transformer_blocks.0.attn2x.to_out.0.weight", "output_blocks.3.1.transformer_blocks.0.attn2x.to_out.0.bias", "output_blocks.4.1.transformer_blocks.0.attn1x.to_q.weight", "output_blocks.4.1.transformer_blocks.0.attn1x.to_k.weight", "output_blocks.4.1.transformer_blocks.0.attn1x.to_v.weight", "output_blocks.4.1.transformer_blocks.0.attn1x.to_out.0.weight", "output_blocks.4.1.transformer_blocks.0.attn1x.to_out.0.bias", "output_blocks.4.1.transformer_blocks.0.attn2x.to_q.weight", "output_blocks.4.1.transformer_blocks.0.attn2x.to_k.weight", "output_blocks.4.1.transformer_blocks.0.attn2x.to_v.weight", "output_blocks.4.1.transformer_blocks.0.attn2x.to_out.0.weight", "output_blocks.4.1.transformer_blocks.0.attn2x.to_out.0.bias", "output_blocks.5.1.transformer_blocks.0.attn1x.to_q.weight", "output_blocks.5.1.transformer_blocks.0.attn1x.to_k.weight", "output_blocks.5.1.transformer_blocks.0.attn1x.to_v.weight", "output_blocks.5.1.transformer_blocks.0.attn1x.to_out.0.weight", "output_blocks.5.1.transformer_blocks.0.attn1x.to_out.0.bias", "output_blocks.5.1.transformer_blocks.0.attn2x.to_q.weight", "output_blocks.5.1.transformer_blocks.0.attn2x.to_k.weight", "output_blocks.5.1.transformer_blocks.0.attn2x.to_v.weight", "output_blocks.5.1.transformer_blocks.0.attn2x.to_out.0.weight", "output_blocks.5.1.transformer_blocks.0.attn2x.to_out.0.bias", "output_blocks.6.1.transformer_blocks.0.attn1x.to_q.weight", "output_blocks.6.1.transformer_blocks.0.attn1x.to_k.weight", "output_blocks.6.1.transformer_blocks.0.attn1x.to_v.weight", "output_blocks.6.1.transformer_blocks.0.attn1x.to_out.0.weight", "output_blocks.6.1.transformer_blocks.0.attn1x.to_out.0.bias", "output_blocks.6.1.transformer_blocks.0.attn2x.to_q.weight", "output_blocks.6.1.transformer_blocks.0.attn2x.to_k.weight", "output_blocks.6.1.transformer_blocks.0.attn2x.to_v.weight", "output_blocks.6.1.transformer_blocks.0.attn2x.to_out.0.weight", "output_blocks.6.1.transformer_blocks.0.attn2x.to_out.0.bias", "output_blocks.7.1.transformer_blocks.0.attn1x.to_q.weight", "output_blocks.7.1.transformer_blocks.0.attn1x.to_k.weight", "output_blocks.7.1.transformer_blocks.0.attn1x.to_v.weight", "output_blocks.7.1.transformer_blocks.0.attn1x.to_out.0.weight", "output_blocks.7.1.transformer_blocks.0.attn1x.to_out.0.bias", "output_blocks.7.1.transformer_blocks.0.attn2x.to_q.weight", "output_blocks.7.1.transformer_blocks.0.attn2x.to_k.weight", "output_blocks.7.1.transformer_blocks.0.attn2x.to_v.weight", "output_blocks.7.1.transformer_blocks.0.attn2x.to_out.0.weight", "output_blocks.7.1.transformer_blocks.0.attn2x.to_out.0.bias", "output_blocks.8.1.transformer_blocks.0.attn1x.to_q.weight", "output_blocks.8.1.transformer_blocks.0.attn1x.to_k.weight", "output_blocks.8.1.transformer_blocks.0.attn1x.to_v.weight", "output_blocks.8.1.transformer_blocks.0.attn1x.to_out.0.weight", "output_blocks.8.1.transformer_blocks.0.attn1x.to_out.0.bias", "output_blocks.8.1.transformer_blocks.0.attn2x.to_q.weight", "output_blocks.8.1.transformer_blocks.0.attn2x.to_k.weight", "output_blocks.8.1.transformer_blocks.0.attn2x.to_v.weight", "output_blocks.8.1.transformer_blocks.0.attn2x.to_out.0.weight", "output_blocks.8.1.transformer_blocks.0.attn2x.to_out.0.bias"]