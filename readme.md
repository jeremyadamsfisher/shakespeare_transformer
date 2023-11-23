# Shakespeare Transformer

This is a slightly remixed verion of the Transformer from [Karpathy's video](https://www.youtube.com/watch?v=kCc8FmEb1nY). I wrote it after following along with the video, then rewriting from my notes without consulting the original video or the code. It's derivative and mainly here to assess my own understanding of self-attention. I have extended it recently to learn more about the overall process of training LM's, including:

- Rewriting self-attention with as few tensor operations as possible, mostly through einsum notation
- Training on out-of-core datasets; currently working on a character-languange model training on all of wikipedia
- Distributed training (i.e., multi-GPU)
- Dispatching jobs to remote machines on vast.ai for scale

It does have some nice features:

- `einops` for rearranging tensors and shape assertions throughout the codebase to clarify the dimensions of everything
- pytorch lightning to keep things organized
- wandb integration for experimental tracking
- hydra for configuration

Here are some example outputs (keep in mind this is a character model, so producing cohert words and grammar is one of the important/non-trivial regularities):

> during wwi, intel salvage television retained its proper overnight show, due to abundance of television networks. at www 260, mccarthy added the forced television service to the rocket. mccarthy had considered bringing pinoy to promotion this vegetarian movement, and decided to merge it as the weather curtailer. coach and mccarthy's team in both 1972 and 1973 were both executives.
>
> in 1975, mccarthy produced a "temporary window," which was not affected by the adventure. during wwmx's budget, bristol rockets added the sound defects and attempted to concentrate on events on serving budgets on the u.s. mediterranean. pinoy, which was also one of the most important television networks in the united states, was announced at a dey mad people's conference in august 2010. 
>
> only one year later, mccarthy introduced a ring that was reviewed in kindred talk show live by about five participants.
> 
> sports illustrations
> in the late 1970s, it was revealed that pinoy team acquired a big supply of goods with first air force pinoy.
>
> from a novel to the newspaper's own until 1979, mccarthy property was sold onto federal commissioner's call sign carrying a mobile supply of goods. after also contracting bayonet doors to have goods, mccarthy pressed to be in trouble to have no regular witnesses in the climax. with the security services paid, mccarthy advertised in a 1978 interview with goods news magazine, "it's not hard to get idols? i am not going on any other supply of goods. i'm going on any other issues. it's a lush shooting shoot."
> 
> commissioners wishing to return to pinoy team included his son, duke louis mccarthy, acting as a minor league player and "the coach of the rocket", commander allen "allison" mccarthy, a pinoy's navy officer. on the same day, the only guests from the team were ever captain clark "good-faith writer cals" mccarthy, who served on subsidiary of the same service.
> 
> mccarthy was sold to new york city-based military corporation, which specialized in marking goods for united states navy officers. "i found out i was paid b

Also, check out the blog post where I delve into [all the things I was wrong about regarding self-attention](https://www.jeremyafisher.com/posts/notes-on-self-attention/). Writing this codebase inspired the blog post.

## Boostrapping

You need to run the dataset preprocessing script BEFORE training.

```
python gpt/convert_wikipedia.py --config-fp gpt/conf/data_config/wikipedia_100K.yaml
```

Unless you have access to my bucket, you have to change the `dataset_uri` field to something else. For example, `./wikipedia-100` 

Then, run the training script. You have to specify a model config and a data config

```
python -O gpt/train.py +model_config=baby ++model_config.batch_size=512 data_config=wikipedia_100
```

## To do

- [ ] Perplexity and accuracy evaluation
- [ ] Implement training schedule from the [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf) paper (appendix, section B)
- [ ] Researchy stuff like [ROPE embeddings](https://paperswithcode.com/method/rope) and [multi-query attention](https://paperswithcode.com/method/multi-query-attention)