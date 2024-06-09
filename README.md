# build nanoGPT

This repo holds the reproduction of [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master), with step-by-step git commits, and additionally there is a video lecture on YouTube where you can see me build it all from scratch and explain the pieces along the way.

We basically start from an empty file and work our way to a reproduction of the GPT-2 (124M) model. This model probably trained for quite some time back in the day when GPT-2 came out (2019, ~5 years ago), but today, reproducing it is a matter of ~1hr and ~$10. Also Note that GPT-2 and GPT-3 and both simple language models, trained on internet document, and they just "dream" internet documents. So this does not cover Chat finetuning, and you can't talk to it like you can talk to ChatGPT. The finetuning process comes after this part and will be covered later. For now this is the kind of stuff that the 124M model says if you prompt it with "Hello, I'm a language model," after 10B tokens of training:

```
Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone, and to find out the different grammar rules
Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff.
Hello, I'm a language model, and the question is, what should I do if I want to be a teacher?
Hello, I'm a language model, and I'm an English person. In languages, "speak" is really speaking. Because for most people, there's
```

Lol. Anyway, once the video comes out, this will also be a place for FAQ, and a place for fixes and errata, of which I am sure there will be a number :)

For discussions and questions, please use [Discussions tab](https://github.com/karpathy/build-nanogpt/discussions), and for faster communication, have a look at my [Zero To Hero Discord](https://discord.gg/3zy8kqD9Cp):

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## Video

[![Build nanoGPT Video Lecture]()

## Errata

## Run

## FAQ

## License

MIT
