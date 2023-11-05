# Shakespeare Transformer

## To do

- [x] Reproducible training with bumpmyversion/docker
- [ ] Resumable training
- [x] BPE/sentencepiece tokenization
- [x] Using all tokens in the dataset
- [ ] Evaluation

This is a slightly remixed verion of the Transformer from [Karpathy's video](https://www.youtube.com/watch?v=kCc8FmEb1nY). I wrote it after following along with the video, then rewriting from my notes without consulting the original video or the code. It's derivative and mainly here to assess my own understanding of self-attention. I have extended it recently to learn more about the overall process of training LM's.

It does have some nice features:

- `einops` for rearranging tensors and shape assertions throughout the codebase to clarify the dimensions of everything
- pytorch lightning to keep things organized
- wandb integration for experimental tracking

Also, check out the blog post where I delve into [all the things I was wrong about regarding self-attention](https://www.jeremyafisher.com/posts/notes-on-self-attention/). Writing this codebase inspired the blog post.
