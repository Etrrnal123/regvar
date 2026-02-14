import torch
from torch import nn

def mlp(in_dim, hidden_mults, out_dim):
    layers = []
    dim = in_dim
    for m in hidden_mults:
        layers.append(nn.Linear(dim, int(in_dim * m)))
        layers.append(nn.ReLU(inplace=True))
        dim = int(in_dim * m)
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)

class SAINT(nn.Module):
    def __init__(
        self,
        num_continuous,
        dim,
        dim_out,
        depth,
        heads,
        attn_dropout,
        ff_dropout,
        mlp_hidden_mults,
        cont_embeddings,
        attentiontype,
        final_mlp_style,
        y_dim,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.dim = dim
        self.simple_MLP = nn.ModuleList([mlp(1, (2,), dim) for _ in range(num_continuous)])
        self.register_buffer("con_mask_offset", torch.arange(num_continuous).long())
        self.mask_embeds_cont = nn.Embedding(num_continuous * 2 + 2, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=attn_dropout, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=max(1, depth))
        self.pt_mlp = mlp(num_continuous * dim, (2,), num_continuous * dim_out)
        self.pt_mlp2 = mlp(num_continuous * dim, (2,), num_continuous * dim_out)
        self.rec_head = nn.Linear(dim, 1)

    def transformer(self, x):
        return self.transformer_enc(x)

    def forward(self, x_cont_enc):
        out = self.rec_head(x_cont_enc).squeeze(-1)
        return [out]
