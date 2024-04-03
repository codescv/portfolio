---
title: Lessons Learned Developing Text Ads Generation
summary: The lessons I learned from developing a text ads generation project using language models the past over 3 years.
date: 2024-03-20
tags:
  - NLP
  - DeepLearning
  - Transformer
---

Online advertising plays a pivotal role in driving business growth and brand awareness. Text ads, particularly within the realm of search engine and social media, present a concise and cost-effective method for reaching targeted audiences.  The ability to craft compelling and persuasive ad copy is paramount to maximizing conversion rates and return on investment (ROI).

The text ad generation problem can be formulated as a function, where the input is the context (e.g. product name, keywords, background, audience etc) and the output is the text ad.  The goal is to write diverse, relevant, catchy ad copies with good performance.

# Iteration 1:  T5 Finetuning (Late 2020 - 2021)
I started the project in late 2020s, when the status of text generation was quite different from what it is today. To give some context:
- There were no notions of "LLMs" yet (although GPT-3 was published in June 2020, it was out of the reach for most people). 
- [T5](https://arxiv.org/pdf/1910.10683.pdf)(2019) was (one of) the best text generation model available (considering both quality and cost) back then.
	- The model sizes of T5 are 60M (small), 220M (base), 770M (t5-large), 3B and 11B.
	- There was later (late 2020s) a multi-lingual version of T5 (aka [mt5](https://huggingface.co/google/mt5-base)) supporting 101 languages. 
- No quantization and no float16 as far as I was aware of.
- No efficient fine-tuning
- Model with more than 1B params were quite infeasible for single node prediction.

T5 is an encoder-decoder transformer model, which means you design an input and an output in the training data (as apposed to only "text" in a decoder only model).

Following the T5 [conventions](https://blog.research.google/2020/02/exploring-transfer-learning-with-t5.html) , the training data input was designed as a task name followed by the context:
```
input: Generate ad headline: [KW] {keywords} [Category] {product_category} ...(other attributes)
output: {ad_headline}
```

## Dataset
The first version of the dataset was built by collecting text ads in the real world. However, a lot of cleaning has to be involved:
- There are a lot of duplications.  Imagine how many times you get "x% off" or "Free Shipping" from a retail customer.
- To help the model learn the relationship between the context (keywords) vs the ad text,  some simple algorithms (e.g. token coverage) are used to improve the relevance.
- Formatting problems (there can be a lot)
	- Capitalization
	- White spaces
	- Length too long or too short

## Model
We use T5 for English-only datasets and MT5 for multi-lingual (both the base sized model).  From our experiences the dataset should be a few hundred at minimum, and trained for >10 epochs.

## Post Processing
The model outputs are not perfect and require post processing to be ready to use in campaigns.
- Since the model outputs are sampled independently at prediction time, it's obvious that deduplication is needed.
- The output still suffers from the formatting problems mentioned above. So the same data cleaning process apply.
- We also filter by rules to get rid of non-factual predictions (e.g. predicting "free shipping" for a customer that does not offer it)
- MT5 had slight chance outputting incorrect languages. So we also filter by language detection.


# Iteration 2: LLMs Zero Shot (2022 - 2023)
In 2021 Google built LaMDA(the LLM that pre-existed Bard and Gemini) so we had an internal access to it.  It became obvious that LLMs had a lot of potentials in doing text ad generation in zero shot fashion.

Basically what we did was to use a prompt for all the background information, and just chat with the LLM to get answers back.  It (kind of) worked, with a few catches though:

- **Format control** was bad.  The model can return numbered list, unordered list with or without leading symbols (such as -, dot, etc). There a a lot of different format that the model can return. 
- **Length control** was bad. The model always outputs very length ad copies which exceeds the limit of Google Search Ads, no matter how hard you try prompting it.

That said, these problems are pretty much fixed with Gemini and ChatGPT in around 2023. But at that time(2021-2022) it caused us a lot of trouble. We tried to use a lot of pattern matching initially and later even trained a small T5 network just to extract the ad headlines from the model outputs.

# Iteration 3: Finetuning LLMs (2024)
In 2023-2024 a lot of open sourced LLMs were released.  We also experimented if it's possible to finetune an open source model to achieve similar results than zero shot Gemini.  It turns out to work quite well.  We open sourced our approach as follows:

## Dataset Preparing
[Colab](https://colab.sandbox.google.com/github/google/gps-babel-tower/blob/main/examples/text_ads_generation/01%20-%20Dataset%20Builder.ipynb)
We generated a text ad generation dataset using Gemini.
1. Decide a vertical (rough type) of the customer e.g. E-commerce, Game, Apps etc.
2. Generate categories. e.g. for e-commerce we used this prompt:
```
I'm planning to build an e-commerce website like Amazon. Do the following for me:

1. Generate 20 top level categories of products that I can sell
2. Generate a list of sub categories for each top level categories.

Return the result in yaml format.
```

3. For each category, generate product information and ad copies.

One prompting trick: asking the model to return yaml or json will help you parsing the result.

## SFT
[Colab](https://colab.sandbox.google.com/github/google/gps-babel-tower/blob/main/examples/text_ads_generation/02%20-%20SFT%20Trainer.ipynb)
Fine tuned Gemma-2B on the dataset.

## Reinforcement Learning
[Colab](https://colab.research.google.com/github/google/gps-babel-tower/blob/main/examples/text_ads_generation/04%20-%20PPO.ipynb)
We also tried RLHF to improve the result.  The most important thing for this step is to design the reward. Here are a few ideas:
1. Formatting score (Capitalization, punctuations, no leading / trailing spaces etc)
2. Diversity
3. Performance (click rate, impressions etc)
It showed a little improvement (score from 1.0->1.1), but considering the trouble it takes, it may or may not worth it:
1. RL models are much harder to train than SFT
2. Requires more memory (e.g. need two copies of the model, although if using LoRA they can share the base model)
3. Improvement is quite minimal




