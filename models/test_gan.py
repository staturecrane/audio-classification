import torch

from models.gan import Generator


def test_sanity():
    zeros = torch.zeros(1, 100)
    generator = Generator(64, 100, 1)
    output = generator(zeros)

    assert output.size() == (1, 1, 37533)
