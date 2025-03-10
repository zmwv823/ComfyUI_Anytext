import torch
import numpy as np
import os
from  comfy import model_management as mm
from comfy.model_patcher import ModelPatcher
from comfy.latent_formats import SDXL, SD15, Flux
from comfy.cli_args import args
from PIL import Image
import gc
import folder_paths
import cv2
import logging

logger = logging.getLogger("Anytext")

should_offload = torch.cuda.is_available() or torch.mps.is_available() or torch.xpu.is_available()

device = mm.get_torch_device()

def tensor2numpy_cv2(tensor_img):
    arr_img = tensor_img.numpy()[0] * 255
    arr_img = arr_img.astype(np.uint8)
    return arr_img
    
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def numpy_cv2tensor(img):
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img)[None,]
    return img

def get_device_by_name(device, debug: bool=False):
    """
    Args:
        "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta", "directml"],{"default": "auto"}), 
    """
    if device == 'auto':
        try:
            # device = "cpu"
            # if torch.cuda.is_available():
            #     device = "cuda"
            #     # device = torch.device("cuda")
            # elif torch.backends.mps.is_available():
            #     device = "mps"
            #     # device = torch.device("mps")
            # elif torch.xpu.is_available():
            #     device = "xpu"
            #     # device = torch.device("xpu")
            device = mm.get_torch_device()
        except:
                raise AttributeError("What's your device(到底用什么设备跑的)？")
    # elif device == 'cuda':
    #     device = torch.device("cuda")
    # elif device == "mps":
    #     device = torch.device("mps")
    # elif device == "xpu":
    #     device = torch.device("xpu")
    if debug:
        print("\033[93mUse Device(使用设备):", device, "\033[0m")
    return device

def get_dtype_by_name(dtype, debug: bool=False):
    """
    "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto"}),返回模型精度选择。
    """
    if dtype == 'auto':
        try:
            if mm.should_use_fp16():
                dtype = torch.float16
            elif mm.should_use_bf16():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtypes manually.")
    elif dtype== "fp16":
         dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype = torch.float32
    elif dtype == "fp8_e4m3fn":
        dtype = torch.float8_e4m3fn
    elif dtype == "fp8_e4m3fnuz":
        dtype = torch.float8_e4m3fnuz
    elif dtype == "fp8_e5m2":
        dtype = torch.float8_e5m2
    elif dtype == "fp8_e5m2fnuz":
        dtype = torch.float8_e5m2fnuz
    if debug:
        print("\033[93mModel Precision(模型精度):", dtype, "\033[0m")
    return dtype
        
def clean_up(debug=False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()
    else: 
        if debug:
            print('\033[93m', 'Not needed', '\033[0m')
        pass

def get_files_with_extension(folder_name, extension=['.safetensors']):
    """_summary_

    Args:
        folder_name (_type_): models根目录下的文件夹，不能指向更深的子目录。
        extension (list, optional): 要获取的文件的后缀。

    Returns:
        _type_: _description_
    """
    try:
        folders = folder_paths.get_folder_paths(folder_name)
    except:
        folders = []

    if not folders:
        folders = [os.path.join(folder_paths.models_dir, folder_name)]
    if not os.path.isdir(folders[0]):
        folders = [os.path.join(folder_paths.base_path, folder_name)]
    if not os.path.isdir(folders[0]):
        return {}
    
    filtered_folders = []
    for x in folders:
        if not os.path.isdir(x):
            continue
        the_same = False
        for y in filtered_folders:
            if os.path.samefile(x, y):
                the_same = True
                break
        if not the_same:
            filtered_folders.append(x)

    if not filtered_folders:
        return {}

    output = {}
    for x in filtered_folders:
        files, folders_all = folder_paths.recursive_search(x, excluded_dir_names=[".git"])
        filtered_files = folder_paths.filter_files_extensions(files, extension)

        for f in filtered_files:
            output[f] = x

    return output

def download_singlefile_from_huggingface(repo_id, file_name, root_folder=None, new_name=None, revision="main"):
    """_summary_

    Args:
        function: 从huggingface下载单个文件到指定文件夹或者.cache，无需提前创建根目录文件夹
        repo_id (_type_): repoid,例如模型地址https://huggingface.co/Kijai/depth-fm-pruned，那么repoid就是Kijai/depth-fm-pruned
        file_name (_type_): 要下载的文件名字，包含后缀
        root_folder (_type_): 下载到哪个目录下，如果为None，则缓存到.cache
        new_name (_type_, optional): 是否要重命名，如果重命名，这个就是新名字，包含后缀
        revision: 分支
    """
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id, file_name, local_dir=root_folder, revision=revision)
    if new_name != None and root_folder != None:
        old_file_name = os.path.join(root_folder, file_name)
        new_file_name = os.path.join(root_folder, new_name)
        os.rename(old_file_name, new_file_name)

def download_repoid_model_from_huggingface(repo_id: str, Base_Path: str, ignore_patterns: list = None, resume_download: bool=False):
    """_summary_

    Args:
        function: 根据提供的repo名字下载文件夹模型内全部文件到指定的模型目录(非.cache)，无需提前创建目录，纯离线。
        repo_id (str): 模型ID，作者名字+模型名字,repo_id，例如模型网址https://huggingface.co/ZhengPeng7/BiRefNet_lite，那么repo_id就是ZhengPeng7/BiRefNet_lite。
        Base_Path (str): 下载到本地指定目录(非根目录，要模型文件夹目录)(例如D:\AI\ComfyUI_windows_portable\ComfyUI\models\rembg\models--ZhengPeng7--BiRefNet_lite)。
        local_dir_use_symlinks:是否使用blob编码存放，这种存放方式，第一次执行加载时会需要全球网络连接huggingface查找更新。
        ignore_patterns: ["*x*"(指定文件夹), "*.x"(指定后缀的文件)]不下载某些文件，例如snapshot_download(repo_id="lemonaddie/geowizard", ignore_patterns=["*vae*", "*.ckpt", "*.pt", "*.png", "*non_ema*", "*safety_checker*", "*.bin"], 忽略下载safety_checker、vae和safety_checker整个文件夹、以及其他后缀的文件.ckpt---.pt---.png---.bin。
    Returns:
        无
    """
    from huggingface_hub import snapshot_download as hg_snapshot_download
    hg_snapshot_download(repo_id, 
                        local_dir=Base_Path, 
                        ignore_patterns=ignore_patterns, 
                        local_dir_use_symlinks=False,
                        resume_download=resume_download, 
                    )

Pillow_Color_Names = [
    "red",
    "green",
    "blue",
    "white",
    "black",
    "yellow",
    "pink",
    "gold",
    "purple",
    "brown",
    "orange",
    "tomato",
    "violet",
    "wheat",
    "snow",
    "yellowgreen",
    "gray",
    "grey",
    "aliceblue",
    "antiquewhite",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "blanchedalmond",
    "blueviolet",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgray",
    "darkgrey",
    "darkgreen",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "darkslategrey",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dimgray",
    "dimgrey",
    "dodgerblue",
    "firebrick",
    "floralwhite",
    "forestgreen",
    "fuchsia",
    "gainsboro",
    "ghostwhite",
    "goldenrod",
    "greenyellow",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgreen",
    "lightgray",
    "lightgrey",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "lightskyblue",
    "lightslategray",
    "lightslategrey",
    "lightsteelblue",
    "lightyellow",
    "lime",
    "limegreen",
    "linen",
    "magenta",
    "maroon",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "navy",
    "oldlace",
    "olive",
    "olivedrab",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "plum",
    "powderblue",
    "rebeccapurple",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "silver",
    "skyblue",
    "slateblue",
    "slategray",
    "slategrey",
    "springgreen",
    "steelblue",
    "tan",
    "teal",
    "thistle",
    "turquoise",
    "whitesmoke",
    ]

def cv2img_canny(img, low_threshold=64, high_threshold=100):
    """_summary_

    Args:
        img (_type_): 输入cv2_numpy类型图片，从tensor转换需要numpy_cv2tensor
        low_threshold (int, optional): _description_. Defaults to 64.
        high_threshold (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: 输出pillow(np.array)类型图片，转换成tensor需要pil2tensor
    """
    # low_threshold = 64
    # high_threshold = 100
    img = cv2.Canny(img, low_threshold, high_threshold)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    return Image.fromarray(img)

Scheduler_Names = ["Euler", "Euler Karras", "Euler a", "DPM++ 2M", "DPM++ 2M Karras", "DPM++ 2M SDE", "DPM++ 2M SDE Karras", "DDIM", "UniPC", "UniPC Karras", "DPM++ SDE", "DPM++ SDE Karras", "DPM2", "DPM2 Karras", "DPM2 a", "DPM2 a Karras", "Heun", "Heun Karras", "LMS", "LMS Karras", "TCD", "LCM", "PNDM", "DEIS", "DEIS Karras", "DDPM", 'SASolver']#, "IPNDM"]
        
def SD15_Scheduler_List():
    """_summary_

    Args:
        scheduler(model_default_scheduler): 
            from diffusers import DDIMScheduler
            scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

    Returns:
        _type_: SD15 Scheduler List
    """
    from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, UniPCMultistepScheduler, DDIMScheduler, TCDScheduler, LCMScheduler, PNDMScheduler, DEISMultistepScheduler, DDPMScheduler, SASolverScheduler#, IPNDMScheduler
    from .sd15_scheduler_configs import ddim_config, pndm_config, lcm_config, deis_config, tcd_config
    schedulers = {
            "Euler": EulerDiscreteScheduler.from_config(pndm_config),
            "Euler Karras": EulerDiscreteScheduler.from_config(pndm_config, use_karras_sigmas=True),
            "Euler a": EulerAncestralDiscreteScheduler.from_config(pndm_config),
            "DPM++ 2M": DPMSolverMultistepScheduler.from_config(pndm_config),
            "DPM++ 2M Karras": DPMSolverMultistepScheduler.from_config(pndm_config, use_karras_sigmas=True),
            "DPM++ 2M SDE": DPMSolverMultistepScheduler.from_config(pndm_config, algorithm_type="sde-dpmsolver++"),
            "DPM++ 2M SDE Karras": DPMSolverMultistepScheduler.from_config(pndm_config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"),
            "DDIM": DDIMScheduler.from_config(ddim_config),
            "UniPC": UniPCMultistepScheduler.from_config(pndm_config),
            "UniPC Karras": UniPCMultistepScheduler.from_config(pndm_config, use_karras_sigmas=True),
            "DPM++ SDE": DPMSolverSinglestepScheduler.from_config(pndm_config, lower_order_final=True),
            "DPM++ SDE Karras": DPMSolverSinglestepScheduler.from_config(pndm_config, use_karras_sigmas=True, lower_order_final=True),
            "DPM2": KDPM2DiscreteScheduler.from_config(pndm_config),
            "DPM2 Karras": KDPM2DiscreteScheduler.from_config(pndm_config, use_karras_sigmas=True),
            "DPM2 a": KDPM2AncestralDiscreteScheduler.from_config(pndm_config),
            "DPM2 a Karras": KDPM2AncestralDiscreteScheduler.from_config(pndm_config, use_karras_sigmas=True),
            "Heun": HeunDiscreteScheduler.from_config(pndm_config),
            "Heun Karras": HeunDiscreteScheduler.from_config(pndm_config, use_karras_sigmas=True),
            "LMS": LMSDiscreteScheduler.from_config(lcm_config),
            "LMS Karras": LMSDiscreteScheduler.from_config(lcm_config, use_karras_sigmas=True),
            "TCD": TCDScheduler.from_config(tcd_config),
            "LCM": LCMScheduler.from_config(lcm_config),
            "PNDM": PNDMScheduler.from_config(pndm_config),
            "DEIS": DEISMultistepScheduler.from_config(deis_config),
            "DEIS Karras": DEISMultistepScheduler.from_config(deis_config, use_karras_sigmas=True),
            "DDPM": DDPMScheduler.from_config(pndm_config),
            'SASolver': SASolverScheduler.from_config(pndm_config)
            # "IPNDM": IPNDMScheduler.from_config(lcm_config),
        }
    return schedulers

def SDXL_Scheduler_List():
    from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, UniPCMultistepScheduler, DDIMScheduler, TCDScheduler, LCMScheduler, PNDMScheduler, DEISMultistepScheduler, DDPMScheduler, SASolverScheduler#, IPNDMScheduler
    from .sdxl_scheduler_configs import ddim_config_xl, euler_config_xl
    schedulers = {
            "Euler": EulerDiscreteScheduler.from_config(euler_config_xl),
            "Euler Karras": EulerDiscreteScheduler.from_config(euler_config_xl, use_karras_sigmas=True),
            "Euler a": EulerAncestralDiscreteScheduler.from_config(euler_config_xl, steps_offset=1),
            "DPM++ 2M": DPMSolverMultistepScheduler.from_config(euler_config_xl),
            "DPM++ 2M Karras": DPMSolverMultistepScheduler.from_config(euler_config_xl, use_karras_sigmas=True),
            "DPM++ 2M SDE": DPMSolverMultistepScheduler.from_config(euler_config_xl, algorithm_type="sde-dpmsolver++"),
            "DPM++ 2M SDE Karras": DPMSolverMultistepScheduler.from_config(euler_config_xl, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"),
            "DDIM": DDIMScheduler.from_config(ddim_config_xl),
            "UniPC": UniPCMultistepScheduler.from_config(euler_config_xl, ),
            "UniPC Karras": UniPCMultistepScheduler.from_config(euler_config_xl, use_karras_sigmas=True),
            "DPM++ SDE": DPMSolverSinglestepScheduler.from_config(euler_config_xl, ),
            "DPM++ SDE Karras": DPMSolverSinglestepScheduler.from_config(euler_config_xl, use_karras_sigmas=True),
            "DPM2": KDPM2DiscreteScheduler.from_config(euler_config_xl, ),
            "DPM2 Karras": KDPM2DiscreteScheduler.from_config(euler_config_xl, use_karras_sigmas=True),
            "DPM2 a": KDPM2AncestralDiscreteScheduler.from_config(euler_config_xl, ),
            "DPM2 a Karras": KDPM2AncestralDiscreteScheduler.from_config(euler_config_xl, use_karras_sigmas=True),
            "Heun": HeunDiscreteScheduler.from_config(euler_config_xl, ),
            "Heun Karras": HeunDiscreteScheduler.from_config(euler_config_xl, use_karras_sigmas=True),
            "LMS": LMSDiscreteScheduler.from_config(euler_config_xl, ),
            "LMS Karras": LMSDiscreteScheduler.from_config(euler_config_xl, use_karras_sigmas=True),
            "TCD": TCDScheduler.from_config(ddim_config_xl),
            "LCM": LCMScheduler.from_config(ddim_config_xl),
            "PNDM": PNDMScheduler.from_config(euler_config_xl, ),
            "DEIS": DEISMultistepScheduler.from_config(euler_config_xl, ),
            "DEIS Karras": DEISMultistepScheduler.from_config(euler_config_xl, use_karras_sigmas=True),
            "DDPM": DDPMScheduler.from_config(euler_config_xl, ),
            'SASolver': SASolverScheduler.from_config(euler_config_xl, ),
            # "IPNDM": IPNDMScheduler(),
        }
    return schedulers

def cv2pil(cv2_img):
    """
        OpenCV转换成PIL.Image RGB格式.
    """
    img = Image.fromarray(cv2.cvtColor(cv2_img,cv2.COLOR_BGR2RGB))
    return img

def seperate_masks(mask, sort_priority, gap=102):
    """
    从一张多个不重叠遮罩图中分离多个遮罩，排序位置不是很准确，后期坐标需要重新按x_min、y_min值由小到大排序。
    Args:
        mask: 输入cv2掩膜二值图。
        sort_priority: 排序方向 '↔' ，水平或者垂直 '↕' 。
        gap (int, optional): _description_. Defaults to 102.

    Returns:
        sorted_components: 输出多个单mask图，类型为list，例如np.array(sorted_components[0])就是一张二值图。
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    components = []
    for label in range(1, num_labels):
        component = np.zeros_like(mask)
        component[labels == label] = 255
        components.append((component, centroids[label]))
    if sort_priority == '↕':
        fir, sec = 1, 0  # top-down first
    elif sort_priority == '↔':
        fir, sec = 0, 1  # left-right first
    components.sort(key=lambda c: (c[1][fir]//gap, c[1][sec]//gap))
    sorted_components = [c[0] for c in components]
    return sorted_components

def padding_image(pil_img, max_size=1024):
    """
    图片按最长边缩放到指定尺寸
    Args:
        pil_img: _description_
        max_size: 最长边缩放到尺寸
    Returns:
        _type_: pil_img
    """
    ratio = max_size / max(pil_img.size)
    new_size = tuple([int(x * ratio) for x in pil_img.size])
    img = pil_img.resize(new_size, Image.LANCZOS)
    return img

def comfy_clean_vram(**kwargs):
    loaded_models = mm.loaded_models()
    if kwargs.get("model") in loaded_models:
        print(" - Model found in memory, unloading...")
        loaded_models.remove(kwargs.get("model"))
    else:
        # Just delete it, I need the VRAM!
        model = kwargs.get("model")
        if type(model) == dict:
            keys = [(key, type(value).__name__) for key, value in model.items()]
            for key, model_type in keys:
                if key == 'model':
                    print(f"Unloading model of type {model_type}")
                    del model[key]
                    # Emptying the cache after this should free the memory.
    mm.free_memory(1e30, mm.get_torch_device(), loaded_models)
    mm.soft_empty_cache(True)
    try:
        print(" - Clearing Cache...")
        clean_up()
    except:
        print("   - Unable to clear cache")
    #time.sleep(2) # why?
    return (list(kwargs.values()))

latent_format_map = {
    "SD15": SD15(),
    "SDXL": SDXL(),
    "Flux": Flux(),
}

class CustomModelPatcher(ModelPatcher):
    """
        Overrall model or pipeline or main model 整体模型或者管线或者主模型
            upper of model.diffusion_model(unet_upper), 
            unet, clip, vae = model.model.diffusion_model, model.cond_stage_model, model.first_stage_model
            clip_text_model, tokenizer = model.first_stage_model.transformer, model.first_stage_model.tokenizer
    为class添加属性：Base_Model().__setattr__("tokenizer", None)
    打印属性：attributes = dir(Base_Model) -> dict
    for attr in attributes: 
        if not attr.startswith('__'): 
            print(f"{attr} = {getattr(Base_Model, attr)}")
    """
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        super().__init__(model, load_device, offload_device, size, weight_inplace_update)
        """
        """
        self.model = model #ldm overrall model that must contain a state_dict, model.diffusion_model
        
        self.sub_models = []
        self.sub_models_noSD = []
        
        self.sub_patchers = []
        
        self.unet = None
        self.unet_encoder = None
        self.vae = None
        self.vae1 = None
        self.clip = None
        self.tokenizer = None
        self.clip_vision = None
        self.clip_processor = None
        self.image_processor = None
        self.feature_extractor = None
        self.processor = None
        
        self.embedding = None
        self.control_net = None
        self.ip_adapter = None
        
        self.visual_tokenizer = None
        
        self.model_type = None
        
        self.device = None
        self.dtype = None
        
        self.quantization = "None"
        self.load_kwargs = None
        
        #lora modules
        self.model_clone = None
        self.clip_clone = None
        
        # hook modelClass.device setter
        def set_value(self, new_value):
            pass
        
        try:
            self.model.__class__.device = property(fget=self.model.__class__.device.fget, fset=set_value) #must be torch.nn.Module that have class(device)
        except Exception as e:
            # logger.error(e)
            logger.warning(f'Warning: {e}')
    
    def set_latent_preview(self, model_type: str="SD15"):
        """
        For comfy latent_preview:
            model_type (str, optional): _description_. Defaults to "SD15", supported types: SD15、SDXL、Flux.
            load_device (torch.DeviceObjType, optional): _description_. Defaults to mm.get_torch_device().
        """
        self.model.__setattr__("latent_format", latent_format_map[model_type])
        
    def set_submodels_device_value(self):
        # hook modelClass.device setter
        def set_value(self, new_value):
            pass
        for k in self.sub_models: #sub models, must be torch.nn.Module that have class(device)
                k.__setattr__("set_value", set_value)
                try:
                    k.__class__.device = property(fget=k.__class__.device.fget, fset=set_value)
                except Exception as e:
                    # logger.error(e)
                    logger.warning(f'Warning: {e}')
        
    def check_inpaint(self, isInpaint: bool=False):
        if isInpaint:
            self.__setattr__("isInpaint", isInpaint)
            
    def offload_or_unload(self,
                        keep_load: bool=True,
                        keep_device: bool=True,
                    ):
        """
        模型管理
        """
        if keep_load and keep_device and should_offload and self.device.type != "cpu" and not args.gpu_only and self.quantization != "8-bit":
            # logger.info(f"Offload model: {self.model_type} to cpu (ram) on {self.load_device.type} device.")
            logger.info(f"Offload model: {self.model_type} to cpu (ram) on {self.load_device.type} device.")
            torch.nn.ModuleList(self.sub_models + [self.model]).to(self.offload_device)
            mm.soft_empty_cache(True)
            clean_up()
            self.device = self.offload_device
        elif not keep_load:
            for i in self.sub_models_noSD+self.sub_models+self.sub_patchers:
                del i
            self.model = self.unet = self.unet_encoder = self.vae = self.clip = self.tokenizer = self.clip_vision = self.clip_processor = self.image_processor = self.feature_extractor = self.processor = self.embedding = self.control_net = self.ip_adapter = self.visual_tokenizer = self.model_clone = self.clip_clone = None
            self.sub_models = self.sub_models_noSD = self.sub_patchers = []
            logger.info(f"Try unload model: {self.model_type} to release resources on {self.load_device.type} device.")
            mm.soft_empty_cache(True)
            clean_up()
            
    def load_all_gpu(self, force_cpu: bool=False, to_load: bool=True, force_load: bool=False):
        if should_offload and self.quantization != "8-bit":
            if (not force_cpu and self.device.type == "cpu") or force_load:
                logger.info(f'Load model: {self.model_type} to gpu.')
                mm.load_models_gpu([self] + self.sub_patchers, force_full_load=True)
                if self.quantization != "8-bit" and to_load:
                    torch.nn.ModuleList(self.sub_models + [self.model]).to(self.load_device)
                self.device = self.load_device
            elif force_cpu and self.device.type != "cpu":
                logger.info(f'Offload model: {self.model_type} to cpu with force_cpu checked.')
                torch.nn.ModuleList(self.sub_models + [self.model]).to(self.offload_device)
                mm.soft_empty_cache(True)
                clean_up()
                self.device = self.offload_device
            
    def custom_load(
        self,
        models: list=[],
        patchers: list=[],
        load_device: torch.DeviceObjType=mm.get_torch_device(),
        offload_device: torch.DeviceObjType=mm.text_encoder_offload_device(),
        force_cpu: bool=False,
        force_load: bool=True
    ):
        """
        Custom load models:
            patchers (list, optional): _description_. Defaults to [].
            models (list, optional): _description_. Defaults to [].
            offload_device (torch.DeviceObjType, optional): _description_. Defaults to mm.text_encoder_offload_device().
            force_cpu (bool, optional): _description_. Defaults to False.
        """
        load_device = load_device if load_device != self.load_device else self.load_device
        offload_device = offload_device if offload_device != self.offload_device else self.offload_device
        if should_offload:
            if (not force_cpu and self.device.type == "cpu") or force_load:
                logger.info(f'Load mdoels to gpu.')
                torch.nn.ModuleList(models).to(load_device)
                if len(patchers) != 0:
                    mm.load_models_gpu(patchers)
            elif force_cpu and self.device.type != "cpu":
                logger.info("Offload model: load_device=='cuda' to cpu (ram) with force_cpu checked.")
                torch.nn.ModuleList(models).to(offload_device)
                mm.soft_empty_cache(True)
                clean_up()
        
    def custom_offload(
        self,
        models: list=[],
        do_clean: bool=False,
        offload_device: torch.DeviceObjType=mm.text_encoder_offload_device(),
        force_offload: bool=True
    ):
        offload_device = offload_device if offload_device != self.offload_device else self.offload_device
        if should_offload:
            if (not args.gpu_only and self.device.type != "cpu") or force_offload:
                logger.info(f"Offload models to cpu (ram) on device with cuda (vram), do_clean: {do_clean}.")
                torch.nn.ModuleList(models).to(offload_device)
                if do_clean:
                    mm.soft_empty_cache(True)
                    clean_up()
    
    def custom_unload(self, models: list=[]):
        for i in models:
            del i
        models = []
        mm.soft_empty_cache(True)
        clean_up()
        
    def forward(self):
        pass
        

class Kwargs:
    """
    为class添加属性：Kwargs().__setattr__("image", None)
    打印属性：attributes = dir(Kwargs) -> dict
    for attr in attributes: 
        if not attr.startswith('__'): 
            print(f"{attr} = {getattr(Kwargs, attr)}")
    """
    def __init__(self, kwargs: dict=None):
        if kwargs != None and kwargs != {}:
            for k, v in kwargs.items():
                self.__setattr__(k, v)
                
        self.keep_load = True
        self.keep_device = True
        
    def forward(self):
        pass