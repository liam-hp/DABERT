import torch, math
from torch import nn, cuda
from transformers import BertModel, BertConfig

device = "cuda" if cuda.is_available() else "cpu"


class CustomBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        # Replace the attention layer in each transformer
        for layer in self.encoder.layer:
            layer.attention.self = CustomAttention(config)
    

class CustomAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Implement your custom attention mechanism here
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.linlayer = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # Custom attention mechanism
        # Example: simple scaled dot-product attention (much simplified)
        query_layer = self.query(hidden_states)
        value_layer = self.value(hidden_states)
        combined = torch.cat((query_layer, value_layer), 3).to(device)
        output = self.linlayer(combined)
        return output


