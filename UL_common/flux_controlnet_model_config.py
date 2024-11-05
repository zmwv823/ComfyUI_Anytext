Flux_InstantX_ControlNet_config = {
  "_class_name": "FluxControlNetModel",
  "_diffusers_version": "0.30.0.dev0",
  "_name_or_path": "/mnt/wangqixun/",
  "attention_head_dim": 128,
  "axes_dims_rope": [
    16,
    56,
    56
  ],
  "guidance_embeds": True,
  "in_channels": 64,
  "joint_attention_dim": 4096,
  "num_attention_heads": 24,
  "num_layers": 5,
  "num_mode": 10,
  "num_single_layers": 10,
  "patch_size": 1,
  "pooled_projection_dim": 768
}

Flux_XLabs_ControlNet_config = {
  "_class_name": "FluxControlNetModel",
  "_diffusers_version": "0.31.0.dev0",
  "attention_head_dim": 128,
  "axes_dims_rope": [
    16,
    56,
    56
  ],
  "guidance_embeds": True,
  "in_channels": 64,
  "conditioning_embedding_channels": 16,
  "joint_attention_dim": 4096,
  "num_attention_heads": 24,
  "num_layers": 2,
  # "num_mode": null,
  "num_single_layers": 0,
  "patch_size": 2,
  "pooled_projection_dim": 768
}

HunyuanDit_ControlNet_config = {
  "_class_name": "HunyuanDiT2DControlNetModel",
  "_diffusers_version": "0.30.0.dev0",
  "activation_fn": "gelu-approximate",
  "attention_head_dim": 88,
  "conditioning_channels": 3,
  "cross_attention_dim": 1024,
  "cross_attention_dim_t5": 2048,
  "hidden_size": 1408,
  "in_channels": 4,
  "learn_sigma": True,
  "mlp_ratio": 4.3637,
  "norm_type": "layer_norm",
  "num_attention_heads": 16,
  "num_layers": 40,
  "patch_size": 2,
  "pooled_projection_dim": 1024,
  "sample_size": 128,
  "text_len": 77,
  "text_len_t5": 256,
  "transformer_num_layers": 40,
  "use_style_cond_and_image_meta_size": False
}