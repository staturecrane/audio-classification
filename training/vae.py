import argparse

import torch
import torchaudio

from datasets.audio import get_audio_dataset
from models.vae import Encoder, Decoder


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--data-directory",
    type=str,
    required=True,
    help="Directory where subfolders of audio reside",
)

parser.add_argument("-e", "--num-epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("-b", "--batch-size", type=int, default=50, help="Batch size")


def main(data_directory, num_epochs, batch_size):
    dataset = get_audio_dataset(
        data_directory, max_length_in_seconds=7, pad_and_truncate=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8
    )

    train_dataloader_len = len(dataloader)

    encoder = Encoder(64, 1, 100).to("cuda")
    decoder = Decoder(64, 1, 100).to("cuda")

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        for sample_idx, (audio, _) in enumerate(dataloader):
            batch_size = audio.size(0)
            decoder.zero_grad()
            encoder.zero_grad()

            audio = audio.to("cuda")
            # train with real
            z, mu, logvar = encoder(audio)
            decoded = decoder(z)
            decoded = decoded.narrow(2, 0, 32000)

            mse = criterion(decoded, audio)

            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            loss = mse + KLD
            loss.backward()

            optimizer_encoder.step()
            optimizer_decoder.step()

            print(
                f"{epoch:06d}-[{sample_idx + 1}/{train_dataloader_len}]: loss {loss.mean().item()}"
            )

            if sample_idx % 100 == 0:
                with torch.no_grad():
                    fake_noise = torch.randn(1, 100, 1).to("cuda")
                    output_gen = decoder(fake_noise).narrow(2, 0, 112_000).to("cpu")
                    torchaudio.save(
                        f"outputs/decoder_output_{epoch:06d}_{sample_idx:06d}.wav",
                        output_gen[0],
                        16000,
                    )
        torch.save(generator.state_dict(), "%s/netG_epoch_%d.pth" % (opt.outf, epoch))
        torch.save(
            discriminator.state_dict(), "%s/netD_epoch_%d.pth" % (opt.outf, epoch)
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
