import torch

from models.gan import Generator


def test_sanity():
    zeros = torch.zeros(1, 100, 1)
    generator = Generator(1, 100)
    output = generator(zeros)

    assert output.size() == (1, 1, 16530)
