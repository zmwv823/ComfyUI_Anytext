SDXL_ControlNet_Union_xinsir_v10_promax_config = { # xinsir--controlnet-union-sdxl-1.0_promax.json
        "_class_name": "ControlNetModel",
        "_diffusers_version": "0.20.0.dev0",
        "act_fn": "silu",
        "addition_embed_type": "text_time",
        "addition_embed_type_num_heads": 64,
        "addition_time_embed_dim": 256,
        "attention_head_dim": [
            5,
            10,
            20
        ],
        "block_out_channels": [
            320,
            640,
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
        "cross_attention_dim": 2048,
        "down_block_types": [
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D"
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
        "projection_class_embeddings_input_dim": 2816,
        "resnet_time_scale_shift": "default",
        "transformer_layers_per_block": [
            1,
            2,
            10
        ],
        # "upcast_attention": None,
        "use_linear_projection": True,
        "num_control_type": 8
    }

SDXL_ControlNet_config = {
  "_class_name": "ControlNetModel",
  "_diffusers_version": "0.20.0.dev0",
  "_name_or_path": "valhalla/depth-2",
  "act_fn": "silu",
  "addition_embed_type": "text_time",
  "addition_embed_type_num_heads": 64,
  "addition_time_embed_dim": 256,
  "attention_head_dim": [
    5,
    10,
    20
  ],
  "block_out_channels": [
    320,
    640,
    1280
  ],
#   "class_embed_type": null,
  "conditioning_channels": 3,
  "conditioning_embedding_out_channels": [
    16,
    32,
    96,
    256
  ],
  "controlnet_conditioning_channel_order": "rgb",
  "cross_attention_dim": 2048,
  "down_block_types": [
    "DownBlock2D",
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D"
  ],
  "downsample_padding": 1,
#   "encoder_hid_dim": null,
#   "encoder_hid_dim_type": null,
  "flip_sin_to_cos": True,
  "freq_shift": 0,
  "global_pool_conditions": False,
  "in_channels": 4,
  "layers_per_block": 2,
  "mid_block_scale_factor": 1,
  "norm_eps": 1e-05,
  "norm_num_groups": 32,
#   "num_attention_heads": null,
#   "num_class_embeds": null,
  "only_cross_attention": False,
  "projection_class_embeddings_input_dim": 2816,
  "resnet_time_scale_shift": "default",
  "transformer_layers_per_block": [
    1,
    2,
    10
  ],
#   "upcast_attention": null,
  "use_linear_projection": True
}