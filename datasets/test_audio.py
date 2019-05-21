from datasets.audio import get_audio_dataset


def test_sanity():
    dataset = get_audio_dataset("test")

    print(dataset[0][1])
    assert False
