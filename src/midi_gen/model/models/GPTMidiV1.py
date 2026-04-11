from torch import nn

from midi_gen.model.models.TransformerBlock import TransformerBlock
from midi_gen.data_management.tokenizing import create_vocabulary

class GPTMidiV1(nn.Module):

    def __init__(self, vocab_len = None, d_model=512, num_heads=8, num_layers=6, ff_dim_ratio=4, dropout=0.1):
        super().__init__()
        if vocab_len is None:
            vocabulary, _ = create_vocabulary()
            vocab_len = len(vocabulary)

        self.embedding = nn.Embedding(vocab_len, d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, ff_dim_ratio=ff_dim_ratio, dropout=dropout, causal=True)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)

        self.out_proj = nn.Linear(d_model, vocab_len, bias=False)
        self.out_proj.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.layer_norm(x)
        logits = self.out_proj(x)
        return logits



