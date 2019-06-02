import argparse

import torch
import torchaudio

from datasets.audio import get_audio_dataset
from models.music_gan import Discriminator, Generator
from models.vae import Encoder


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
        data_directory, max_length_in_seconds=2, pad_and_truncate=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8
    )

    train_dataloader_len = len(dataloader)

    encoder = Encoder(64, 1, 100).to("cuda")
    generator = Generator(64, 1, 100).to("cuda")
    discriminator = Discriminator(64, 1).to("cuda")

    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    criterion = torch.nn.BCELoss()
    MSE = torch.nn.MSELoss()

    real_label = 1
    fake_label = 0

    for epoch in range(num_epochs):
        for sample_idx, (audio, _) in enumerate(dataloader):
            batch_size = audio.size(0)

            encoder.zero_grad()
            generator.zero_grad()
            discriminator.zero_grad()

            audio = audio.to("cuda")

            z, mu, logvar = encoder(audio)
            decoded = generator(z)
            decoded = decoded.narrow(2, 0, 32000)

            mse = MSE(decoded, audio)

            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            vae_loss = mse + KLD
            vae_loss.backward()

            optimizer_enc.step()
            optimizer_gen.step()

            generator.zero_grad()

            label = torch.full((batch_size,), real_label).to("cuda")

            # train with real
            disc_output_real = discriminator(audio)
            err_d_real = criterion(disc_output_real, label)
            err_d_real.backward()

            # train with fake
            noise = torch.randn(batch_size, 100, 1).to("cuda")
            fake = generator(noise)
            fake = fake.narrow(2, 0, 32000)
            label.fill_(fake_label)

            disc_output_fake = discriminator(fake.detach())
            err_d_fake = criterion(disc_output_fake, label)
            err_d_fake.backward()

            err_d = err_d_real + err_d_fake
            optimizer_disc.step()

            # generator training
            generator.zero_grad()
            label.fill_(real_label)
            gen_output = discriminator(fake)
            err_g = criterion(gen_output, label)
            err_g.backward()

            optimizer_gen.step()

            print(
                f"{epoch:06d}-[{sample_idx + 1}/{train_dataloader_len}]: disc_loss: {err_d.mean().item()}, gen_loss: {err_g.mean().item()}, vae_loss: {vae_loss.mean().item()}"
            )

            if sample_idx % 100 == 0:
                with torch.no_grad():
                    fake_noise = torch.randn(1, 100, 1).to("cuda")
                    output_gen = generator(fake_noise).narrow(2, 0, 32000).to("cpu")
                    torchaudio.save(
                        f"outputs/generator_output_{epoch:06d}_{sample_idx:06d}.wav",
                        output_gen[0],
                        16000,
                    )

        torch.save(
            encoder.state_dict(), "%s/encoder_epoch_%d.pth" % ("checkpoints", epoch)
        )
        torch.save(
            generator.state_dict(), "%s/generator_epoch_%d.pth" % ("checkpoints", epoch)
        )
        torch.save(
            discriminator.state_dict(),
            "%s/discrimiantor_epoch_%d.pth" % ("checkpoints", epoch),
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
