import torch.nn as nn
from einops import rearrange
import torch


## Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1, max_len: int = 197):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, hidden_dim))
        
    def forward(self):
        x = self.pos_embed
        return self.dropout(x)


## Path Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_dim= 768, out_dim = 768, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(in_features = in_dim, out_features = out_dim)
        
    def forward(self, x):
        x = self.proj(x)
        return self.dropout(x)

## Feedforward Linear layer in the attention layer
class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class AttentionLayer(nn.Module):
    def __init__(self, embed_dim = 768, n_heads = 8, dropout = 0.1):

        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim%n_heads == 0, f'embed_dim: {embed_dim} is not a multiple of n_heads:{n_heads}'
        self.head_dim = embed_dim // n_heads

        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        key_vector = rearrange(self.fc_k(x), 'b t (head k) -> head b t k', head=self.n_heads)
        query_vector = rearrange(self.fc_q(x), 'b l (head k) -> head b l k', head=self.n_heads)
        value_vector = rearrange(self.fc_v(x), 'b t (head k) -> head b t k', head=self.n_heads)
        
        attn_score = torch.einsum('hblk,hbtk->hblt', query_vector, value_vector)*self.scale
        attn_probs = self.dropout(torch.softmax(attn_score, dim=-1))

        context_vector = torch.einsum('hblt,hbtv->hblv', attn_probs, value_vector)
        final_context_vector = rearrange(context_vector, 'head b t d -> b t (head d)')
        return self.to_out(final_context_vector)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormAttn(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))



class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([])
        for _ in range(config['num_hidden_layers']):
            encoder_block = nn.ModuleList([
                            PreNormAttn(config['hidden_size'],
                                        AttentionLayer(config['hidden_size'],
                                                                config['num_attention_heads'],
                                                                config['hidden_dropout_prob'],
                                                                )
                                        ),
                            PreNorm(config['hidden_size'],
                                    FeedForward(config['hidden_size'],
                                                config['hidden_size'] * config['intermediate_ff_size_factor'],
                                                dropout=config['hidden_dropout_prob']))
                        ])
            self.layers.append(encoder_block)

    def forward(self,x):
        for attn, ff in self.layers:
          skip = x + attn(x)
          x = ff(skip) + skip
          
        return x


class VisionTransformer(nn.Module):
      def __init__(self, config):
            super().__init__()
            self.encoder_block = Encoder(config)

            ## Learnable parameter
            self.cls_token =  nn.Parameter(torch.zeros(1, config['hidden_size']))
            
            self.pe = PositionalEncoding(hidden_dim =config['hidden_size'],max_len =  config['max_seq_len']+1)
            self.patch = PatchEmbedding(in_dim =config['total_patches'], out_dim = config['hidden_size'])
        
      def forward(self, x):
            
            x = self.patch(x)
            x = torch.cat([self.cls_token.expand(x.shape[0], 1, -1), x], axis = 1)
            x = self.pe() + x
            x = self.encoder_block(x)
            return x