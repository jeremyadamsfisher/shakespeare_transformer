# Shakespeare Transformer

This is a slightly remixed verion of the Transformer from [Karpathy's video](https://www.youtube.com/watch?v=kCc8FmEb1nY). I wrote it after following along with the video, then rewriting from my notes without consulting the original video or the code. It's derivative and mainly here to assess my own understanding of self-attention. I have extended it recently to learn more about the overall process of training LM's, including:

- Training on out-of-core datasets; currently working on a character-languange model trained on 75K articles wikipedia (approx. 500M tokens/characters)
- Dispatching jobs to remote machines on vast.ai

It does have some nice features:

- `einops` for rearranging tensors and shape assertions throughout the codebase to clarify the dimensions of everything
- pytorch lightning to keep things organized
- wandb integration for experimental tracking

Here are some example outputs (keep in mind this is a character model, so producing cohert words and grammar is one of the important/non-trivial regularities)

> year and 1993 cpus). however, new york city consti 
> fierceptochemistry, protein welling procedure awak 
> vowel of students "to quickly desire 90,000 to vow 

Also, check out the blog post where I delve into [all the things I was wrong about regarding self-attention](https://www.jeremyafisher.com/posts/notes-on-self-attention/). Writing this codebase inspired the blog post.


## To do

- [ ] Resumable training
- [ ] Perplexity and accuracy evaluation
- [ ] Implement training schedule from the [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf) paper (appendix, section B)