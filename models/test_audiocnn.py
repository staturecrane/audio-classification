from models.audiocnn import AudioCNN
from test.fixtures import audio_tensor


def test_sanity(audio_tensor):
    audio, _ = audio_tensor
    audio_cnn = AudioCNN(1)

    output = audio_cnn.forward(audio.unsqueeze(0))

    assert output.size() == (1, 1)
