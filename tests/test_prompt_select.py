from src import generate as G
from src.utils import load_config


def run(overrides):
    seen = {}

    def fake_chat(cfg, model, messages, fmt, cache):
        seen["system"] = messages[0]["content"]
        return '{"claims": [], "insufficient_evidence": true}'

    orig, G._chat = G._chat, fake_chat
    try:
        G.generate_claims(load_config("config/config.yaml", overrides=overrides), "m", "q", "[S1] text", None)
    finally:
        G._chat = orig
    return seen["system"]


def test_default_claims_prompt_is_unchanged():
    assert "one atomic claim per item" in run({})


def test_claims_prompt_can_be_switched_to_v2():
    assert "restates ONE sentence of ONE passage" in run({"claims_prompt": "system_claims_v2.txt"})