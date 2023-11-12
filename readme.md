# Shakespeare Transformer

This is a slightly remixed verion of the Transformer from [Karpathy's video](https://www.youtube.com/watch?v=kCc8FmEb1nY). I wrote it after following along with the video, then rewriting from my notes without consulting the original video or the code. It's derivative and mainly here to assess my own understanding of self-attention. I have extended it recently to learn more about the overall process of training LM's, including:

- Rewriting self-attention with as few tensor operations as possible, mostly through einsum notation
- Training on out-of-core datasets; currently working on a character-languange model training on all of wikipedia
- Dispatching jobs to remote machines on vast.ai for scale

It does have some nice features:

- `einops` for rearranging tensors and shape assertions throughout the codebase to clarify the dimensions of everything
- pytorch lightning to keep things organized
- wandb integration for experimental tracking
- hydra for configuration

Here are some example outputs (keep in mind this is a character model, so producing cohert words and grammar is one of the important/non-trivial regularities):

> alexander ii, italian-american people's last daugh
> there is a natural history that can be still shut
> obligations into modern infancy that would be abl

Also, check out the blog post where I delve into [all the things I was wrong about regarding self-attention](https://www.jeremyafisher.com/posts/notes-on-self-attention/). Writing this codebase inspired the blog post.


## To do

- [ ] Perplexity and accuracy evaluation
- [ ] Implement training schedule from the [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf) paper (appendix, section B)
- [ ] Researchy stuff like [ROPE embeddings](https://paperswithcode.com/method/rope) and [multi-query attention](https://paperswithcode.com/method/multi-query-attention)