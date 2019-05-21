import pytest
import torchaudio


@pytest.fixture
def audio_tensor():
    return torchaudio.load("test/data/test.wav", normalization=True)
