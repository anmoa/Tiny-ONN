import json
from pathlib import Path
from typing import ClassVar, Dict, List

class ArcChatMLTokenizer:
    SPECIAL_TOKENS: ClassVar[Dict[str, str]] = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "eos_token": "[EOS]",
        "im_start": "<|im_start|>",
        "im_end": "<|im_end|>",
    }

    ROLES: ClassVar[List[str]] = ["system", "problem", "scratchpad", "solution"]
    
    def __init__(self, vocab_path: Path | str | None = None):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        
        if vocab_path and Path(vocab_path).exists():
            self.load_vocab(vocab_path)
        else:
            self._build_vocab()

    def _build_vocab(self):
        token_id = 0
        
        # Color tokens
        for i in range(10):
            token_str = str(i)
            self.vocab[token_str] = token_id
            self.inverse_vocab[token_id] = token_str
            token_id += 1

        # Special tokens
        for token_str in self.SPECIAL_TOKENS.values():
            self.vocab[token_str] = token_id
            self.inverse_vocab[token_id] = token_str
            token_id += 1
            
        # Role tokens
        for role_str in self.ROLES:
            self.vocab[role_str] = token_id
            self.inverse_vocab[token_id] = token_str
            token_id += 1

    def save_vocab(self, vocab_path: Path | str):
        Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)
            
    def load_vocab(self, vocab_path: Path | str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return self.vocab_size

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(token, self.vocab[self.SPECIAL_TOKENS["unk_token"]]) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.inverse_vocab.get(id, self.SPECIAL_TOKENS["unk_token"]) for id in ids]

    def encode_grid_with_role(self, grid: torch.Tensor, role: str) -> List[int]:
        if role not in self.ROLES:
            raise ValueError(f"Unknown role: {role}. Available roles: {self.ROLES}")

        grid_tokens = [str(pixel.item()) for pixel in grid.flatten()]
        
        message_tokens = (
            [self.SPECIAL_TOKENS["im_start"]]
            + [role]
            + grid_tokens
            + [self.SPECIAL_TOKENS["im_end"]]
        )
        
        return self.convert_tokens_to_ids(message_tokens)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.SPECIAL_TOKENS["pad_token"]]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.SPECIAL_TOKENS["eos_token"]]
