import argparse

import torch

from datasets.audio import get_audio_dataset
from models.audiocnn import AudioCNN


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

    dataset_length = len(dataset)
    train_length = round(dataset_length * 0.8)
    test_length = dataset_length - train_length

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(train_length), int(test_length)]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=5, num_workers=4
    )

    train_dataloader_len = len(train_dataloader)
    test_dataloader_len = len(test_dataloader)

    audio_cnn = AudioCNN(len(dataset.classes)).to("cuda")

    cross_entropy = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(audio_cnn.parameters())

    for epoch in range(num_epochs):
        audio_cnn.train()
        for sample_idx, (audio, target) in enumerate(train_dataloader):
            audio_cnn.zero_grad()
            audio, target = audio.to("cuda"), target.to("cuda")

            output = audio_cnn(audio)
            loss = cross_entropy(output, target)

            loss.backward()
            optimizer.step()

            print(
                f"{epoch:06d}-[{sample_idx + 1}/{train_dataloader_len}]: {loss.mean().item()}"
            )

        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for sample_idx, (audio, target) in enumerate(test_dataloader):
                audio, target = audio.to("cuda"), target.to("cuda")

                output = audio_cnn(audio)
                test_loss += cross_entropy(output, target)

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            print(f"Evaluation loss: {test_loss.mean().item() / test_dataloader_len}")
            print(f"Evaluation accuracy: {100 * correct / total}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
