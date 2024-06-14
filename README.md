# GPT2 with Native Audio Generation

- run `python sherlock.py`
- run `python train_gpt2.py`
- Fix MultiGPU (not sure whats broken, maybe dataset is too chunked)

[Colab notebook](https://colab.research.google.com/drive/1n05pnDYuBVIyB3HlKzBaoyjIWta7HqG-?usp=sharing)
Note when using a t4 torch.compile doesnt work and use lower batch size

```
# for colab
total_batch_size = 2048*16
B = 2 # micro batch size
T = 2048 # sequence length
```

# Samples

https://github.com/nivibilla/build-nanogpt/assets/26687662/ed05fdff-d117-4222-b0dd-d069252f5324



https://github.com/nivibilla/build-nanogpt/assets/26687662/62518181-93dd-4024-baa5-4a359db657e7



https://github.com/nivibilla/build-nanogpt/assets/26687662/35eadeca-0a29-49af-91f7-9d7114915754



https://github.com/nivibilla/build-nanogpt/assets/26687662/428ac644-e029-438a-9ef6-a5c895a55b7d



https://github.com/nivibilla/build-nanogpt/assets/26687662/1f447b57-07ed-4921-a4d9-29b238efc1b7



https://github.com/nivibilla/build-nanogpt/assets/26687662/36712b34-0661-4eb9-957c-bbe27ae3670d



https://github.com/nivibilla/build-nanogpt/assets/26687662/cabb016b-5e4f-4176-9d21-c1723b59912a



https://github.com/nivibilla/build-nanogpt/assets/26687662/e998f25b-f4fd-4e59-bf15-2079e5e534d5

