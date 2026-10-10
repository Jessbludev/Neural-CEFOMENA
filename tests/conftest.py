import pytest
import torch

from cefomena.config import PRESETS
from cefomena.model.model import CefomenaLM
from cefomena.tokenizer.bpe import BPETokenizer
from cefomena.train.data import iter_texts
from cefomena.train.sources import write_jsonl
from cefomena.train.synth import synthetic_conversations


@pytest.fixture(scope="session")
def synth_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "synth.jsonl"
    write_jsonl(synthetic_conversations(300, seed=1), path)
    return path


@pytest.fixture(scope="session")
def tokenizer(synth_path):
    return BPETokenizer.train(iter_texts([synth_path]), vocab_size=PRESETS["test"].vocab_size)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return CefomenaLM(PRESETS["test"]).eval()
