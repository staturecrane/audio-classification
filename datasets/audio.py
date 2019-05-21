from functools import partial

import torch
import torchaudio

from torchvision.datasets import DatasetFolder


def audio_loader(path, max_length_in_seconds, pad_and_truncate):
    audio_tensor, sample_rate = torchaudio.load(path, normalization=True)
    max_length = sample_rate * max_length_in_seconds
    audio_size = audio_tensor.size()

    if pad_and_truncate:
        if audio_size[1] < max_length:
            difference = max_length - audio_size[1]
            padding = torch.zeros(audio_size[0], difference)
            padded_audio = torch.cat([audio_tensor, padding], 1)
+            return padded_audio

        if audio_tensor.size(1) > max_length:
            return audio_tensor[:max_length]
    return audio_tensor


def get_audio_dataset(datafolder, max_length_in_seconds=2, pad_and_truncate=False):
    loader_func = partial(
        audio_loader,
        max_length_in_seconds=max_length_in_seconds,
        pad_and_truncate=pad_and_truncate,
    )
    dataset = DatasetFolder(datafolder, loader_func, ".wav")

    return dataset
