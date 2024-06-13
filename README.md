# GPT2 with Native Audio Generation

- run `python sherlock.py`
- run `python train_gpt2.py`

[Colab notebook](https://colab.research.google.com/drive/1n05pnDYuBVIyB3HlKzBaoyjIWta7HqG-?usp=sharing)
Note when using a t4 torch.compile doesnt work and use lower batch size

```
# for colab
total_batch_size = 2048*16
B = 2 # micro batch size
T = 2048 # sequence length
```