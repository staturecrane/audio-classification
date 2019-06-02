import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary
    If batch shuffle is enabled, only a single shuffle is applied to the entire
    batch, rather than each sample in the batch.
    """

    def __init__(self, shift_factor, batch_shuffle=False):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor
        self.batch_shuffle = batch_shuffle

    def forward(self, x):
        # Return x if phase shift is disabled
        if self.shift_factor == 0:
            return x

        if self.batch_shuffle:
            # Make sure to use PyTorcTrueh to generate number RNG state is all shared
            k = (
                int(torch.Tensor(1).random_(0, 2 * self.shift_factor + 1))
                - self.shift_factor
            )

            # Return if no phase shift
            if k == 0:
                return x

            # Slice feature dimension
            if k > 0:
                x_trunc = x[:, :, :-k]
                pad = (k, 0)
            else:
                x_trunc = x[:, :, -k:]
                pad = (0, -k)

            # Reflection padding
            x_shuffle = F.pad(x_trunc, pad, mode="reflect")

        else:
            # Generate shifts for each sample in the batch
            k_list = (
                torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1)
                - self.shift_factor
            )
            k_list = k_list.numpy().astype(int)

            # Combine sample indices into lists so that less shuffle operations
            # need to be performed
            k_map = {}
            for idx, k in enumerate(k_list):
                k = int(k)
                if k not in k_map:
                    k_map[k] = []
                k_map[k].append(idx)

            # Make a copy of x for our output
            x_shuffle = x.clone()

            # Apply shuffle to each sample
            for k, idxs in k_map.items():
                if k > 0:
                    x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode="reflect")
                else:
                    x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode="reflect")

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle


class Discriminator(nn.Module):
    def __init__(self, num_dec_features, num_channels):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(num_channels, num_dec_features, 25, 5, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(2),
            nn.Conv1d(num_dec_features, num_dec_features * 2, 25, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(2),
            nn.Conv1d(num_dec_features * 2, num_dec_features * 4, 25, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(2),
            nn.Conv1d(num_dec_features * 4, num_dec_features * 8, 25, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(2),
            nn.Conv1d(
                num_dec_features * 8, num_dec_features * 16, 25, 4, 1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(2),
            nn.Conv1d(num_dec_features * 16, 1, 20, 4, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        output = self.main(features)
        return self.sigmoid(output.view(features.size(0)))


class Generator(nn.Module):
    def __init__(self, num_enc_features, num_channels, num_z, post_proc_filt_len=512):
        super(Generator, self).__init__()
        self.num_enc_features = num_enc_features
        self.post_proc_filt_len = post_proc_filt_len
        self.main = nn.Sequential(
            nn.ConvTranspose1d(num_z, num_enc_features * 16, 25, 4, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose1d(
                num_enc_features * 16, num_enc_features * 8, 25, 4, 1, bias=False
            ),
            nn.ReLU(True),
            nn.ConvTranspose1d(
                num_enc_features * 8, num_enc_features * 4, 25, 4, 1, bias=False
            ),
            nn.ReLU(True),
            nn.ConvTranspose1d(
                num_enc_features * 4, num_enc_features * 2, 25, 4, 1, bias=False
            ),
            nn.ReLU(True),
            nn.ConvTranspose1d(
                num_enc_features * 2, num_enc_features, 25, 4, 1, bias=False
            ),
            nn.ReLU(True),
            nn.ConvTranspose1d(num_enc_features, num_channels, 25, 5, 1, bias=False),
            nn.Tanh(),
        )

        self.ppfilter1 = nn.Conv1d(num_channels, num_channels, post_proc_filt_len)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, tensor):
        output = self.main(tensor)
        if (self.post_proc_filt_len % 2) == 0:
            pad_left = self.post_proc_filt_len // 2
            pad_right = pad_left - 1
        else:
            pad_left = (self.post_proc_filt_len - 1) // 2
            pad_right = pad_left
        output = self.ppfilter1(F.pad(output, (pad_left, pad_right)))

        return output
