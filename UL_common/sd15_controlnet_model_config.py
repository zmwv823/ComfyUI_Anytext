SD15_ControlNet_Canny_JoyType_v11M_config = {
  "_class_name": "ControlNetModel",
  "_diffusers_version": "0.21.1",
  "act_fn": "silu",
#   "addition_embed_type": None,
  "addition_embed_type_num_heads": 64,
#   "addition_time_embed_dim": None,
  "attention_head_dim": 8,
  "block_out_channels": [
    320,
    640,
    1280,
    1280
  ],
#   "class_embed_type": None,
  "conditioning_channels": 3,
  "conditioning_embedding_out_channels": [
    16,
    32,
    96,
    256
  ],
  "controlnet_conditioning_channel_order": "rgb",
  "cross_attention_dim": 768,
  "down_block_types": [
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "DownBlock2D"
  ],
  "downsample_padding": 1,
#   "encoder_hid_dim": None,
#   "encoder_hid_dim_type": None,
  "flip_sin_to_cos": True,
  "freq_shift": 0,
  "global_pool_conditions": False,
  "in_channels": 4,
  "layers_per_block": 2,
  "mid_block_scale_factor": 1,
  "norm_eps": 1e-05,
  "norm_num_groups": 32,
#   "num_attention_heads": None,
#   "num_class_embeds": None,
  "only_cross_attention": False,
#   "projection_class_embeddings_input_dim": None,
  "resnet_time_scale_shift": "default",
  "transformer_layers_per_block": 1,
  "upcast_attention": False,
  "use_linear_projection": False
}

SD15_ControlNet_config = {
  "_class_name": "ControlNetModel",
  "_diffusers_version": "0.16.0.dev0",
  "_name_or_path": "/home/patrick/controlnet_v1_1/control_v11p_sd15",
  "act_fn": "silu",
  "attention_head_dim": 8,
  "block_out_channels": [
    320,
    640,
    1280,
    1280
  ],
  # "class_embed_type": null,
  "conditioning_embedding_out_channels": [
    16,
    32,
    96,
    256
  ],
  "controlnet_conditioning_channel_order": "rgb",
  "cross_attention_dim": 768,
  "down_block_types": [
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "DownBlock2D"
  ],
  "downsample_padding": 1,
  "flip_sin_to_cos": True,
  "freq_shift": 0,
  "in_channels": 4,
  "layers_per_block": 2,
  "mid_block_scale_factor": 1,
  "norm_eps": 1e-05,
  "norm_num_groups": 32,
  # "num_class_embeds": null,
  "only_cross_attention": False,
  # "projection_class_embeddings_input_dim": null,
  "resnet_time_scale_shift": "default",
  "upcast_attention": False,
  "use_linear_projection": False
}

SD15_ControlNet_config = {
  "_class_name": "ControlNetModel",
  "_diffusers_version": "0.14.0.dev0",
  "act_fn": "silu",
  "attention_head_dim": 8,
  "block_out_channels": [
    320,
    640,
    1280,
    1280
  ],
  # "class_embed_type": null,
  "conditioning_embedding_out_channels": [
    16,
    32,
    96,
    256
  ],
  "controlnet_conditioning_channel_order": "rgb",
  "cross_attention_dim": 768,
  "down_block_types": [
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "DownBlock2D"
  ],
  "downsample_padding": 1,
  "flip_sin_to_cos": True,
  "freq_shift": 0,
  "in_channels": 4,
  "layers_per_block": 2,
  "mid_block_scale_factor": 1,
  "norm_eps": 1e-05,
  "norm_num_groups": 32,
  # "num_class_embeds": null,
  "only_cross_attention": False,
  # "projection_class_embeddings_input_dim": null,
  "resnet_time_scale_shift": "default",
  "upcast_attention": False,
  "use_linear_projection": False
}