import torch

from models.music_gan import Generator


def test_sanity():
    zeros = torch.zeros(1, 100, 1).to("cuda")
    generator = Generator(64, 1, 100).to("cuda")
    output = generator(zeros)

    assert output.size() == (1, 1, 120143)
