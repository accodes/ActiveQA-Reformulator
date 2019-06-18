# ActiveQA: Active Question Answering
This repo contains code for our paper [Ask the Right Questions: Active Question
 Reformulation with Reinforcement Learning](https://openreview.net/forum?id=S1CChZ-CZ).

*If you use this code for your research, please [cite the paper](#bibtex).*

## Introduction
ActiveQA is an agent that transforms questions online in order to find the best
answers. The agent consists of a Tensorflow model that reformulates questions
and an Answer Selection model. It interacts with an environment that contains
a question-answering system. The agent queries the environment with variants
of a question and calculates a score for the answer against the original
question. The model is trained end-to-end using reinforcement learning.

This version addresses the [SearchQA](https://arxiv.org/abs/1704.05179)
question-answering task, and the environment consists of the Bi-directional
Attention Flow ([BiDAF](https://github.com/allenai/bi-att-flow)) model of
[Seo et al. (2017)](https://openreview.net/forum?id=HJ0UKP9ge&noteId=HJ0UKP9ge).


## Reformulator Inference Generation

We take the checkpoints provided by the original Active QA repository.
Links:
https://storage.cloud.google.com/pretrained_models/translate.ckpt-6156696.zip
https://storage.googleapis.com/pretrained_models/translate.ckpt-1460356.zip

To generate the reformulations of the given question/s we use 
```
python reformulator_inference.py
```
Flask framework API :
```
python server_reformulator_inference.py
```
