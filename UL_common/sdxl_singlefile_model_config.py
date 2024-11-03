SDXL_VAE_fp16_fix_config = { # madebyollin--sdxl-vae-fp16-fix
    "_class_name": "AutoencoderKL",
    "_diffusers_version": "0.18.0.dev0",
    "_name_or_path": ".",
    "act_fn": "silu",
    "block_out_channels": [
        128,
        256,
        512,
        512
    ],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D"
    ],
    "in_channels": 3,
    "latent_channels": 4,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 512,
    "scaling_factor": 0.13025,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D"
    ],
    "force_upcast": False
}

SDXL_Unet_Base_config = {
  "_class_name": "UNet2DConditionModel",
  "_diffusers_version": "0.19.0.dev0",
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
  "center_input_sample": False,
#   "class_embed_type": None,
  "class_embeddings_concat": False,
  "conv_in_kernel": 3,
  "conv_out_kernel": 3,
  "cross_attention_dim": 2048,
#   "cross_attention_norm": None,
  "down_block_types": [
    "DownBlock2D",
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D"
  ],
  "downsample_padding": 1,
  "dual_cross_attention": False,
#   "encoder_hid_dim": None,
#   "encoder_hid_dim_type": None,
  "flip_sin_to_cos": True,
  "freq_shift": 0,
  "in_channels": 4,
  "layers_per_block": 2,
#   "mid_block_only_cross_attention": None,
  "mid_block_scale_factor": 1,
  "mid_block_type": "UNetMidBlock2DCrossAttn",
  "norm_eps": 1e-05,
  "norm_num_groups": 32,
#   "num_attention_heads": None,
#   "num_class_embeds": None,
  "only_cross_attention": False,
  "out_channels": 4,
  "projection_class_embeddings_input_dim": 2816,
  "resnet_out_scale_factor": 1.0,
  "resnet_skip_time_act": False,
  "resnet_time_scale_shift": "default",
  "sample_size": 128,
#   "time_cond_proj_dim": None,
#   "time_embedding_act_fn": None,
#   "time_embedding_dim": None,
  "time_embedding_type": "positional",
#   "timestep_post_act": None,
  "transformer_layers_per_block": [
    1,
    2,
    10
  ],
  "up_block_types": [
    "CrossAttnUpBlock2D",
    "CrossAttnUpBlock2D",
    "UpBlock2D"
  ],
#   "upcast_attention": None,
  "use_linear_projection": True
}

SDXL_Unet_Inpaint_Base_config = { # diffusers--stable-diffusion-xl-1.0-inpainting-0.1
        "_class_name": "UNet2DConditionModel",
        "_diffusers_version": "0.21.0.dev0",
        "_name_or_path": "valhalla/sdxl-inpaint-ema",
        "act_fn": "silu",
        "addition_embed_type": "text_time",
        "addition_embed_type_num_heads": 64,
        "addition_time_embed_dim": 256,
        "attention_head_dim": [
            5,
            10,
            20
        ],
        "attention_type": "default",
        "block_out_channels": [
            320,
            640,
            1280
        ],
        "center_input_sample": False,
        # "class_embed_type": None,
        "class_embeddings_concat": False,
        "conv_in_kernel": 3,
        "conv_out_kernel": 3,
        "cross_attention_dim": 2048,
        # "cross_attention_norm": None,
        "decay": 0.9999,
        "down_block_types": [
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D"
        ],
        "downsample_padding": 1,
        "dual_cross_attention": False,
        # "encoder_hid_dim": None,
        # "encoder_hid_dim_type": None,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "in_channels": 9,
        "inv_gamma": 1.0,
        "layers_per_block": 2,
        # "mid_block_only_cross_attention": None,
        "mid_block_scale_factor": 1,
        "mid_block_type": "UNetMidBlock2DCrossAttn",
        "min_decay": 0.0,
        "norm_eps": 1e-05,
        "norm_num_groups": 32,
        # "num_attention_heads": None,
        # "num_class_embeds": None,
        "only_cross_attention": False,
        "optimization_step": 37000,
        "out_channels": 4,
        "power": 0.6666666666666666,
        "projection_class_embeddings_input_dim": 2816,
        "resnet_out_scale_factor": 1.0,
        "resnet_skip_time_act": False,
        "resnet_time_scale_shift": "default",
        "sample_size": 128,
        # "time_cond_proj_dim": None,
        # "time_embedding_act_fn": None,
        # "time_embedding_dim": None,
        "time_embedding_type": "positional",
        # "timestep_post_act": None,
        "transformer_layers_per_block": [
            1,
            2,
            10
        ],
        "up_block_types": [
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D"
        ],
        # "upcast_attention": None,
        "update_after_step": 0,
        "use_ema_warmup": False,
        "use_linear_projection": True
    }

SDXL_Unet_Inpaint_config_RealVisXL_V40 = { # OzzyGT--RealVisXL_V4.0
        "_class_name": "UNet2DConditionModel",
        "_diffusers_version": "0.28.0.dev0",
        "act_fn": "silu",
        "addition_embed_type": "text_time",
        "addition_embed_type_num_heads": 64,
        "addition_time_embed_dim": 256,
        "attention_head_dim": [
            5,
            10,
            20
        ],
        "attention_type": "default",
        "block_out_channels": [
            320,
            640,
            1280
        ],
        "center_input_sample": False,
        # "class_embed_type": None,
        "class_embeddings_concat": False,
        "conv_in_kernel": 3,
        "conv_out_kernel": 3,
        "cross_attention_dim": 2048,
        # "cross_attention_norm": None,
        "down_block_types": [
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D"
        ],
        "downsample_padding": 1,
        "dropout": 0.0,
        "dual_cross_attention": False,
        # "encoder_hid_dim": None,
        # "encoder_hid_dim_type": None,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "in_channels": 9,
        "layers_per_block": 2,
        # "mid_block_only_cross_attention": None,
        "mid_block_scale_factor": 1,
        "mid_block_type": "UNetMidBlock2DCrossAttn",
        "norm_eps": 1e-05,
        "norm_num_groups": 32,
        # "num_attention_heads": None,
        # "num_class_embeds": None,
        "only_cross_attention": False,
        "out_channels": 4,
        "projection_class_embeddings_input_dim": 2816,
        "resnet_out_scale_factor": 1.0,
        "resnet_skip_time_act": False,
        "resnet_time_scale_shift": "default",
        # "reverse_transformer_layers_per_block": None,
        "sample_size": 128,
        # "time_cond_proj_dim": None,
        # "time_embedding_act_fn": None,
        # "time_embedding_dim": None,
        "time_embedding_type": "positional",
        # "timestep_post_act": None,
        "transformer_layers_per_block": [
            1,
            2,
            10
        ],
        "up_block_types": [
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D"
        ],
        "upcast_attention": False,
        "use_linear_projection": True
    }