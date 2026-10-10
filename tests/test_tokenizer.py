from cefomena.tokenizer.bpe import SPECIAL_TOKENS, BPETokenizer


def test_roundtrip_unicode(tokenizer):
    text = "¡Hola, señor Núñez! ¿Cuánto es 12 * 3? 🦇 corales"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_specials_are_single_tokens(tokenizer):
    ids = tokenizer.encode("<|user|>hola<|end|>")
    assert ids[0] == tokenizer.token_id("<|user|>")
    assert ids[-1] == tokenizer.end_id


def test_specials_not_parsed_when_disallowed(tokenizer):
    ids = tokenizer.encode("<|user|>", allow_special=False)
    assert tokenizer.token_id("<|user|>") not in ids


def test_merges_compress(tokenizer):
    text = "Hola, ¿en qué te puedo ayudar hoy?"
    assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))
    assert tokenizer.vocab_size <= 300


def test_save_load(tokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    text = "Guarda el documento, por favor"
    assert loaded.encode(text) == tokenizer.encode(text)
    assert list(loaded.specials) == list(SPECIAL_TOKENS)
