# Audio Classification and Generation using CNNs in PyTorch

## Installation 

First, you need to pip install `requirements.txt`:

```shell
pip [pip3] install -r requirements.txt
```

You will also need to install [torchaudio](https://github.com/pytorch/audio) (and its prerequisites). **Note**: torchaudio doesn't seem to work with PyTorch 1.1.0 yet, so this repo remains locked to 1.0.0. I will update once this has been fixed.

## Speech Commands

To run the speech commands dataset, first [download](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) it from Google and extract it into your directory of choice. Then you can call

```shell
PYTHONPATH=. python [python3] training/commands.py -d PATH/TO/SPEECH/COMMANDS/FOLDER -b BATCH_SIZE -e EPOCHS
```

Currently, the script makes a random 80/20 split of the data instead of using Google's CSV splits, but this would be easy to modify. 