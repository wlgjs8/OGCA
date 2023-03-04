import torch
from torch import nn
from einops import rearrange
import fvcore.nn.weight_init as weight_init


def build_saliency_module(embed_dim, num_heads):
    return SaliencyModule(embed_dim, num_heads)


class SaliencyModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.25, n_layers=1):
        super().__init__()

        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, dropout) for _ in range(n_layers)])

    def forward(self, object_features, global_features):
        global_features = rearrange(global_features, 'b c h w -> b (h w) c')
        object_features = object_features.unsqueeze(0)

        for decoder_idx, decoder_layer in enumerate(self.decoder_layers):
            object_features = decoder_layer(object_features, global_features, decoder_idx)
        output = object_features
        output = output.squeeze(0)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        self.self_attention = SelfAttention(embed_dim=embed_dim, num_heads=num_heads, in_channels=1024, dropout=dropout, batch_first=True)
        self.self_attention_layer_norm = nn.LayerNorm(embed_dim)

        self.cross_attention = CrossAttention(embed_dim=embed_dim, num_heads=num_heads, obj_in_channels=1024, global_in_channels=1024, dropout=dropout, batch_first=True)
        self.cross_attention_layer_norm = nn.LayerNorm(embed_dim)

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(embed_dim, dropout=dropout)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, object_features, global_features, decoder_idx=None):
        object_features = self.self_attention_layer_norm(object_features)
        self_attention_output, self_attention_weights = self.self_attention(object_features)
        object_output = object_features + self_attention_output

        global_features = self.cross_attention_layer_norm(global_features)
        cross_attention_output, cross_attention_weights = self.cross_attention(object_output, global_features)
        global_output = object_output + cross_attention_output

        output = self.positionwise_feedforward(global_output)
        output = self.ff_layer_norm(output + self.dropout(output))

        return output

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, in_channels, dropout, batch_first=True):
        super().__init__()

        self.w_query = nn.Linear(in_channels, embed_dim)
        self.w_key = nn.Linear(in_channels, embed_dim)
        self.w_value = nn.Linear(in_channels, embed_dim)

        weight_init.c2_xavier_fill(self.w_query)
        weight_init.c2_xavier_fill(self.w_key)
        weight_init.c2_xavier_fill(self.w_value)

        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, object_features):
        query = self.w_query(object_features)
        key = self.w_key(object_features)
        value = self.w_value(object_features)

        return self.self_attention(query, key, value)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, obj_in_channels, global_in_channels, dropout, batch_first=True):
        super().__init__()

        self.w_query = nn.Linear(obj_in_channels, embed_dim)
        self.w_key = nn.Linear(global_in_channels, embed_dim)
        self.w_value = nn.Linear(global_in_channels, embed_dim)

        weight_init.c2_xavier_fill(self.w_query)
        weight_init.c2_xavier_fill(self.w_key)
        weight_init.c2_xavier_fill(self.w_value)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, object_features, global_features):

        query = self.w_query(object_features)
        key = self.w_key(global_features)
        value = self.w_value(global_features)

        return self.cross_attention(query, key, value)

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, hid_dim)
        self.fc_2 = nn.Linear(hid_dim, hid_dim)

        weight_init.c2_xavier_fill(self.fc_1)
        weight_init.c2_xavier_fill(self.fc_2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        
        return x