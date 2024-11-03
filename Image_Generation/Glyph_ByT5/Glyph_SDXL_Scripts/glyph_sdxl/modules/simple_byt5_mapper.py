from diffusers import ModelMixin
import torch.nn as nn

class ByT5Mapper(ModelMixin):
    def __init__(self, byt5_output_dim, sdxl_text_dim):
        super().__init__()
        self.mapper = nn.Sequential(
                nn.LayerNorm(byt5_output_dim),
                nn.Linear(byt5_output_dim, sdxl_text_dim),
                nn.ReLU(),
                nn.Linear(sdxl_text_dim, sdxl_text_dim)
        )
    
    def forward(self, byt5_embedding):
        return self.mapper(byt5_embedding)
    