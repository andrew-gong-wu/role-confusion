"""CPU-only preparation helpers mirroring the upstream demo.

These functions deliberately avoid Torch, Transformers, CUDA, and model
downloads so dataset construction can be checked on any development laptop.
Tokenization and activation extraction still happen on the remote GPU.
"""

from typing import Dict, Iterable, List


ROLES = ("system", "user", "cot", "assistant", "tool")


def render_single_role_gptoss(role: str, content: str) -> str:
    """Wrap content in the gpt-oss Harmony representation for one role."""
    if role in {"system", "developer", "user"}:
        header = f"{role}<|message|>"
    elif role == "cot":
        header = "assistant<|channel|>analysis<|message|>"
    elif role == "assistant":
        header = "assistant<|channel|>final<|message|>"
    elif role == "tool":
        header = "functions. to=assistant<|channel|>commentary<|message|>"
    else:
        raise ValueError(f"Unsupported role: {role}")
    return f"<|start|>{header}{content}<|end|>"


def build_role_variants(texts: Iterable[str]) -> List[Dict[str, object]]:
    """Create identical-content variants for every probe role.

    `prompt_ix` uniquely identifies a rendered sequence. `base_seq_ix` groups
    all role variants made from the same source text and must be used for the
    train/test split to prevent content leakage.
    """
    rows: List[Dict[str, object]] = []
    for base_seq_ix, text in enumerate(texts):
        for role in ROLES:
            rows.append(
                {
                    "base_seq_ix": base_seq_ix,
                    "prompt_ix": len(rows),
                    "role": role,
                    "content": text,
                    "prompt": render_single_role_gptoss(role, text),
                }
            )
    return rows

