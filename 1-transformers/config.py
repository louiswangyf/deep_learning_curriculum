from dataclasses import dataclass

@dataclass
class GPTConfig:
    n_head: int = 12
    n_layer: int = 12
    n_embd: int = 32
    dropout: float = 0.1
    bias: bool = True
    block_size: int = 10
    vocab_size: int = 2000
    
