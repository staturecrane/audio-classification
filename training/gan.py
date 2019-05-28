import argparse

import torch
import torchaudio

from datasets.audio import get_audio_dataset
from models.gan import Discriminator, Generator


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
        data_directory, max_length_in_seconds=1, pad_and_truncate=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4
    )

    train_dataloader_len = len(dataloader)

    generator = Generator(64, 1, 100).to("cuda")
    discriminator = Discriminator(64, 1).to("cuda")

    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    criterion = torch.nn.BCELoss()

    real_label = 1
    fake_label = 0

    for epoch in range(num_epochs):
        for sample_idx, (audio, _) in enumerate(dataloader):
            batch_size = audio.size(0)
            generator.zero_grad()
            discriminator.zero_grad()

            label = torch.full((batch_size,), real_label).to("cuda")

            audio = audio.to("cuda")

            # train with real
            disc_output_real = discriminator(audio)
            err_d_real = criterion(disc_output_real, label)
            err_d_real.backward()

            # train with fake
            noise = torch.randn(batch_size, 100).to("cuda")
            fake = generator(noise)
            fake = fake.narrow(2, 0, 16000)
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
                f"{epoch:06d}-[{sample_idx + 1}/{train_dataloader_len}]: disc_loss: {err_d.mean().item()}, gen_loss: {err_g.mean().item()}"
            )

            if sample_idx % 10 == 0:
                with torch.no_grad():
                    fake_noise = torch.randn(1, 100).to("cuda")
                    output_gen = generator(fake_noise).narrow(2, 0, 16000).to("cpu")
                    torchaudio.save(
                        f"outputs/generator_output_{epoch:06d}_{sample_idx:06d}.wav",
                        output_gen[0],
                        16000,
                    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
