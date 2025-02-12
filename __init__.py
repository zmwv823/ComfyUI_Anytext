import os
import folder_paths

#加载插件前先检查是否在os.listdir里存在自定义目录，没有则自动创建，防止加载节点失败，官方目录可无视。
comfy_temp_dir = folder_paths.get_temp_directory()
custom_node_root_dir = os.path.dirname(os.path.abspath(__file__))

fonts_path = os.path.join(folder_paths.models_dir, 'fonts')
pretrained_configs_dir = os.path.join(custom_node_root_dir, 'assets', 'Pretrained_Configs')

os.makedirs(fonts_path, exist_ok=True) # fonts

# only import if running as a custom node
try:
	pass
except ImportError:
	pass
else:
	NODE_CLASS_MAPPINGS = {}

# Image Generation
  ## AnyText
	from .Image_Generation.AnyText.nodes import NODE_CLASS_MAPPINGS as AnyText_Nodes
	NODE_CLASS_MAPPINGS.update(AnyText_Nodes)
 
  ## Glyph-SDXL
	from .Image_Generation.Glyph_ByT5.nodes import NODE_CLASS_MAPPINGS as Glyph_SDXL_Nodes
	NODE_CLASS_MAPPINGS.update(Glyph_SDXL_Nodes)
 
  ## JoyType
	from .Image_Generation.JoyType.nodes import NODE_CLASS_MAPPINGS as JoyType_Nodes
	NODE_CLASS_MAPPINGS.update(JoyType_Nodes)

# Image Process 
 ## Image_Process Common
	from .Image_Process.Common.nodes import NODE_CLASS_MAPPINGS as UL_Image_Process_Common_Nodes
	NODE_CLASS_MAPPINGS.update(UL_Image_Process_Common_Nodes)

# Data Process 
 ## Translate
	from .Data_Process.nodes import NODE_CLASS_MAPPINGS as Translator_Nodes
	NODE_CLASS_MAPPINGS.update(Translator_Nodes)

# UL common## Common Loader
	from .UL_common.diffusers_nodes import NODE_CLASS_MAPPINGS as UL_common_loader_Nodes
	NODE_CLASS_MAPPINGS.update(UL_common_loader_Nodes)
 
NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS' ,"WEB_DIRECTORY"]
