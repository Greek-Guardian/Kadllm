"""Build a random text benchmark model and synthetic tokenizer from a local config.

By default, shrinks official dimensions to 4 layers/256 hidden/1024 vocabulary, preserves
128-dimensional gated-delta state heads, and uses a minimal unused vision tower.
--original-config preserves the complete official architecture instead.
The saved BF16 weights are shared by all runners; seed alone is not equivalence.
"""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
)


def main():
    p = argparse.ArgumentParser(
        description="Create one shared random checkpoint from an official Qwen3.5 config; no weight download."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--original-config",
        action="store_true",
        help="Preserve all official architecture dimensions, including vision tower",
    )
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    c = json.loads(Path(a.config).read_text())
    t = c["text_config"]
    if not a.original_config:
        t.update(
            vocab_size=1024,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            max_position_embeddings=512,
            eos_token_id=2,
            pad_token_id=0,
            bos_token_id=1,
            mtp_num_hidden_layers=0,
        )
        t["rope_parameters"]["mrope_section"] = [4, 6, 6]
        c["vision_config"].update(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=1,
            out_hidden_size=256,
            num_position_embeddings=16,
            patch_size=2,
        )
        for key, val in dict(
            image_token_id=1000,
            video_token_id=1001,
            vision_start_token_id=1002,
            vision_end_token_id=1003,
        ).items():
            c[key] = val
    config = Qwen3_5Config.from_dict(c)
    torch.manual_seed(20261024)
    with torch.device("cpu"):
        m = Qwen3_5ForConditionalGeneration(config).eval().to(torch.bfloat16)
    m.save_pretrained(out)
    vocab = {f"t{i}": i for i in range(t["vocab_size"])}
    tok = Tokenizer(WordLevel(vocab, unk_token="t3"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="t3",
        pad_token="t0",
        bos_token="t1",
        eos_token="t2",
    )
    tokenizer.save_pretrained(out)
    # Transformers 4.x/vLLM also recognize this stable tokenizer class name.
    tokenizer_config = out / "tokenizer_config.json"
    metadata = json.loads(tokenizer_config.read_text())
    metadata["tokenizer_class"] = "PreTrainedTokenizerFast"
    tokenizer_config.write_text(json.dumps(metadata, indent=2))
    print(
        json.dumps(
            {
                "parameters": sum(p.numel() for p in m.parameters()),
                "original_config": a.original_config,
                "dtype": str(m.dtype),
                "path": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
