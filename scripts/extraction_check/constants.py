#!/usr/bin/env python3
"""Constants for extraction completeness check."""

import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres"
)

# Expected prompt formats in RawActivation
RAW_FORMATS = ["chat", "role_play", "mc_balanced"]

# Expected extraction strategies in Activation
EXTRACTION_STRATEGIES = [
    "chat_mean", "chat_first", "chat_last", "chat_max_norm", "chat_weighted",
    "role_play", "mc_balanced"
]

# Expected hidden dimensions per model
EXPECTED_HIDDEN_DIMS = {
    "meta-llama/Llama-3.2-1B-Instruct": 2048,
    "Qwen/Qwen3-8B": 4096,
    "meta-llama/Llama-2-7b-chat-hf": 4096,
    "openai/gpt-oss-20b": 6144,
}
