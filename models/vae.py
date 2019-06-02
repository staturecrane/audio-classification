import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_dec_features, num_channels, z_dim):
        super(Encoder, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(num_channels, num_dec_features, 25, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(num_dec_features, num_dec_features * 2, 25, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(num_dec_features * 2, num_dec_features * 4, 25, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(num_dec_features * 4, num_dec_features * 8, 25, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                num_dec_features * 8, num_dec_features * 16, 25, 5, 1, bias=False
            ),
        )

        self.mu = nn.Conv1d(num_dec_features * 16, z_dim, 4, 1, bias=False)
        self.logvar = nn.Conv1d(num_dec_features * 16, z_dim, 4, 1, bias=False)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, features):
        hidden = self.main(features)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, num_enc_features, num_channels, num_z, post_proc_filt_len=512):
        super(Decoder, self).__init__()
        self.num_enc_features = num_enc_features
        self.post_proc_filt_len = post_proc_filt_len
        self.main = nn.Sequential(
            nn.ConvTranspose1d(num_z, num_enc_features * 32, 25, 4, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose1d(
                num_enc_features * 32, num_enc_features * 16, 25, 4, 1, bias=False
            ),
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
            nn.ConvTranspose1d(num_enc_features, num_channels, 25, 4, 1, bias=False),
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
