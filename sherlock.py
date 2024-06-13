import torch, torchaudio
from speech_tokenizer import SpeechTokenizer
import numpy as np
from tqdm import tqdm
import itertools
from pathlib import Path
import os

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

def batch_list(lst, batch_size):
    it = iter(lst)
    return iter(lambda: list(itertools.islice(it, batch_size)), [])

Path('./data').mkdir(parents=True, exist_ok=True)

tokenizer = SpeechTokenizer(device=device)

seconds_per_batch = 3
batch_size = 2
print("batch size:", batch_size)

for audio_path in sorted(os.listdir('./adventures_sherlock_holmes_rg_librivox')):
    print("processing: ", audio_path)
    waves = []
    waveform, sample_rate = torchaudio.load(f'./adventures_sherlock_holmes_rg_librivox/{audio_path}', backend='soundfile')

    # Resample to 24kHz if necessary
    if sample_rate != tokenizer.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=tokenizer.sample_rate)
        waveform = resampler(waveform)

    # Convert to mono by averaging the channels if the audio is stereo
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    i = 0
    while 10*(i+1)*tokenizer.sample_rate < waveform.shape[-1]:
        waves.append(waveform[:, tokenizer.sample_rate*seconds_per_batch*i : tokenizer.sample_rate*seconds_per_batch*(i+1)])
        i+=1
    waves.append(waveform[:, tokenizer.sample_rate*seconds_per_batch*i : ])
    
    batches = list(batch_list(waves, batch_size))
    # batches = batch_list(waves, batch_size)

    single_doc = []
    for batch in tqdm(batches[:-1]):
        encoded_batch = tokenizer.encode(batch)
        for x in encoded_batch:
            single_doc.extend(x[:-1])

    if audio_path.split('_')[1] == '01':
        split = 'val'
    else:
        split = 'train'
    np.save(f"./data/sherlock_{split}_{audio_path.split('_')[1]}", single_doc)