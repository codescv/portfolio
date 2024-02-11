---
title: T5论文笔记
summary: T5是一个Encoder-Decoder架构的transformer. 它的训练过程使用了multi-task learning. 通过研究T5论文我们可以知道transformer模型的预训练中许多有趣的细节.
date: 2020-08-04
tags:
  - Post
image:
  caption: "T5"
---

论文地址: [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)

{{% callout note %}}
2023年注:  时间已经过去大约3年, 超大的GPT和PaLM已经变成LLM的主流. 但是这篇文章仍然值得一读, T5仍然是工业界落地的常见选择.
{{% /callout %}}

# 1. Introduction

这一节简单介绍了T5的背景和目标。从图中可以看到, T5是一个text to text model, 直接把任务名和问题一起输入模型，模型就能输出答案。
这样设计的好处是，同一个模型可以用来解决不同的问题，使得multi task learning成为可能。

![](featured.png)

# 2. Setup

这一节主要介绍T5使用的模型和数据。

## 2.1 Model

这一节介绍了T5使用的模型。早期的一些模型基于RNN, 作为例子这里引用了ULMFiT。目前有一种观点认为RNN(包括LSTM)已经过时了，现在必须用Transformer才够潮。我不同意这种观点：如果你比较贫穷，又想pretrain的话，ULMFiT仍然是个很不错的选择，它在比较少的数据和使用单个GPU就能达到很不错的效果。并且Transformer比RNN效果好并不是因为它在原理上更高明，而是在工程上它更容易scale到超大的模型和超大的数据集(也许这和在推荐系统里DNN能击败Xgboost有类似的原因)。如果有人能找到更快训练RNN的方法，也许哪一天RNN也能重新流行起来，事实上Transformer-XL里有一些结构就有RNN的影子。

{{% callout note %}}
2023年注: 在今天, RNN已经逐渐被Transformer所取代, 但是RNN的思想仍然保留. 事实上, 一些efficient Transformer架构中大量采用的共享参数的策略, 其底层思想就可以认为和RNN非常接近.
{{% /callout %}}

T5基于Transformer，和BERT不同的是，BERT只用了Transformer Encoder. 这样的问题导致BERT在做classification和sequence labeling(例如entity recognition)上效果不错，但是在seq2seq上(例如QA，Translation)效果就没那么好。所以在T5里同时使用了encoder和decoder, 并且把所有问题都统一为text to text.

{{% callout note %}}
2023年注: 在今天, 因为LLM和generation task的流行, 以GPT, PaLM, LLaMA代表的decoder-only架构逐渐成为最流行的架构.  我个人认为相比于encoder-decoder,  decoder-only更精简, 需要调的参数更少(例如: 我怎么知道encoder和decoder应该各占多少参数?), 适应性更强一些.
{{% /callout %}}

T5的Transformer和最早的版本只有很小的区别:

1. remove layer norm bias
2. normalization outside residual path (原始transformer是先add(input, output)再norm, 这里变成了先norm(input)再add(input, output), 大概是这个意思。)
3. relative position embedding

同时论文里也指出，他们没有在实验中验证这些区别能对performance产生多大影响。

{{% callout note %}}
2023年注: 在meta最新的[LLaMA]({{< relref "../llama" >}})中也使用pre-normalization和relative position embedding. 可见Transformer这几年其实变化并不大.
{{% /callout %}}

## 2.2 The Colossal Clean Crawled Corpus

这一节介绍T5 pretrain使用的dataset: C4. C4是Common Crawl的一个子集，**只包含英文**的data(使用langdetect判断), 并且使用了一些cleaning, 去掉重复、质量低的文章，把20TB的数据砍到了750GB. 具体处理见论文。

pretrain只包含英文，可能是T5在translation上表现不佳的原因。

## 2.3 Downstream Tasks

这里介绍了finetuning使用的task. 包括GLUE, SuperGLUE, SQuAD, WMT等等。

## 2.4 Input and Output Format

这里介绍了是怎么样把各种不同的任务dataset转成T5所使用的input format.

例如:

1. Translation task, Input 是"translate English to German: That is good.", output是“Das ist gut.”
2. Classification task, 直接output label.

等等.

# 3. Experiments

T5最大的模型有**11B**个参数，这么大的模型不可能反复实验和调参，所以在pretrain之前，T5 team在小模型上做了很多实验，以找出最合适的model和pretrain task. 他们首先建立了一个baseline model，然后在上面应用各种实验，观察finetune的效果有什么变化。

{{% callout note %}}
2023年注: 目前开源的LLaMA已经有7B和13B, 10B左右的模型已经是比较标准的大小了.
{{% /callout %}}

这些实验包括:

1. model architectures (Section 3.2)
2. unsupervised objectives (Section 3.3)
3. pre-training data sets (Section 3.4)
4. transfer approaches (Section 3.5)
5. scaling (Section 3.6)

## 3.1 Baseline

这一节介绍baseline model.

### 3.1.1 Model
实验中使用了encoder-decoder Transformer model. 结构和BERT - base基本相同，所以参数量大约为后者的两倍。

{{% callout note %}}
注意: 这个是用来做实验的model，并不是最终的T5 model
{{% /callout %}}

### 3.1.2 Training

这里介绍了training使用的一些参数, 这节比较有趣的地方在于给出了一些参数例如steps是怎么来的.

seq len = 512, batch size = 128, 使用了packing[^packing]以后，一个batch里大约有$512*128 = 65536 = 2^{16}$个token. 训练的$step = 2^{19}$ , 所以整个training一共看了$2^{35} = 34B$个token. 这个模型仍然是在C4上train的，而34B远远小于C4的token数，所以模型并没有看过重复的data.

[^packing]: 将尽可能多的sequence放进一条样本内(总长度约等于seq len), 以达到节省计算资源的目的.

在pretrain中，他们使用了$learningRate = 1/\sqrt{max(n,k)}$的方式, n为step, k为一个常数$10^4$ . 他们也提到了Howard等人提出的triangular learning rate会比这个效果稍好。顺便说一下可以去看看fast.ai上的讲解，Howard在图像和NLP模型里都用这个方法达到了不错的效果。之所以这里的experiment没有用这种方法是因为他们的一些实验需要training steps是变化的，而triangular需要一开始知道要训练多少步。

在finetune中，训练的$step = 2^{18}$. 因为finetune使用的数据集有大有小，这个step数是一个考虑到各个数据集的tradeoff. 使用了恒定的$learningRate = 0.001$, 然后每隔5000个step记录结果并选择最好的validation checkpoint. 每个task的最好的performance checkpoint是独立选择的。

### 3.1.3 Vocabulary
T5使用了sentencepice作为tokenization的方案。因为最终要用到translation的方案，所以vocabulary也在German, Fresh, Romanian上fit过一遍。最终vocabulary大小为32k. 所以T5应该没法很好的处理中文和日文了，因为vocabuary里没有这些token, pretrain data里也不包含这些语言。

{{% callout note %}}
对于英文之外的语言如中文和日文, 可以用T5的后续模型[mt5](https://github.com/google-research/multilingual-t5).
{{% /callout %}}

### 3.1.4 Unsupervised Objective
最新的研究表明denoising objective相对于left-to-right LM是对pretrain比较有效的。所以T5使用了类似于BERT的objective. 但不同的是，因为T5是一个seq2seq model，所以跟BERT给input上加mask不同，T5是直接在output上做变换。例如:

{{% callout note %}}
2023年笔记: 在今天无论PaLM还是GPT都抛弃了denoising objective 而使用left-to-right LM(也被称为next token prediction).  所以这一点已经不成立了. 可能在数据量和模型都以倍数增长以后, left-to-right LM已经变得足够好了.
{{% /callout %}}

- **Original Text:** Thank you for inviting me to your party last week.
- **Inputs:** Thank you \<x\> me to your party \<y\> week.
- **Ouput**: \<x\> for inviting \<y\> last \<z\>

具体做法是mask掉15%的token, 然后在输出时只输出被还原的token. 因为输出变得很短，这样能减少pretrain中的计算量。

### 3.1.5 Baseline Performance
这里给出了Baseline model上train-finetune的performance, 以及和training from scratch的performance对比。
主要结论就是baseline model train-finetune的效果在几乎所有任务上都要显著比training from scratch好。所以这种seq2seq pretraining的方法看来是work的。

## 3.2 Architectures
这一节就开始在baseline model上作出一些修改，进行各种个样的实验了。首先是Architecture部分。

### 3.2.1 Model Structures
这一节讨论了各种模型结构上的不同。

首先是attention mask. attention mask是用来限制input和output之间的attention weight, 防止输出时"偷看答案"。
这里讨论的mask有三种: fully visible, causal(只能看到之前的) 和 causal with prefix(某个前缀 fully visible).
![](t5-mask.png)


然后是architecture. 这里讨论了三种: encoder-decoder, language model, prefix LM. 这三个其实跟之前的mask是有关系的。

对于encoder-decoder, 因为输出和输入完全没有overlap, 所以encoder可以使用fully visible. decoder在training的时候不应该look into the future, 所以使用了causal. (这个mask应该是加在了decoder的self attention上， 至于decoder和encoder之间的attention应该是fully visible的)

对于language model, 在predict next token的时候不能看到答案，所以要用causal mask. 注意encoder-decoder中的decoder实际上就是language model.

因为language model中一个token只能attend到之前的token, 这会造成在一些任务例如翻译中损失很多信息。所以有了prefix LM来解决这个问题。修改mask让input中的前面一段可以fully visible.

![](lm-arch.png)

论文里也提到了一点，就是prefix LM在classification任务中其实和BERT是几乎等效的，考虑以下例子:

BERT: \[CLS\] I hate pigeons. \[SEP\] My feelings towards pigeons are filled with animosity.  =>   entailment

prefix LM: premise: I hate pigeons. hypothesis: My feelings towards pigeons are filled with animosity.  =>  entailment

这样，对mask的设定，两者是一样的。


{{% callout note %}}
2023年笔记: 在今天大多数LLM使用的都是中间的(看起来最简单最弱的)language model架构了. 这是不是终极答案, 还犹未可知.
{{% /callout %}}

### 3.2.2 Comparing Different Model Structures
对不同的模型进行对比，应该基于类似的参数量和计算量来进行。但是问题在于，参数量和计算量是不成正比的。例如对于(L+L)层的encoder-decoder模型，它的参数量相当于2L层的language model(只有decoder), 但是计算量只相当于L层的language model. 这是因为对于language model，它必须同时对input和output进行计算; 而encoder只算input, decoder只算output. 基于这个原因，在后面比较的时候，同时列出了计算量和参数量。

### 3.2.3 Objectives
这里的objective用了两种，denoising(类似BERT) 和 language model(predict next token).

### 3.2.4 Results
这里按照Architecture, Objective, 参数量，计算量去比较了各种实验的performance(Table 2).

结论:
1. encoder-decoder的效果比language model和prefix LM要好，即使是在参数量一样，计算量减半的情况下。比较神奇的是，即使把encoder和decoder share参数，效果几乎也一样好。
2. denoising效果总是比language model要好。

## 3.3 Unsupervised Objectives
这里探讨了不同的unsupervised objectives的尝试。

### 3.3.1 Disparate High-Level Approaches
从结果上来看：BERT style会比prefix LM和deshuffling要好，deshuffling是最差的。

### 3.3.2 Simplifying the BERT Objective
这里又细分比较了BERT style的不同做法。其实看起来都差不多，最好的一种方法是replace corrupted spans. 这种方法在CoLA上有奇效，可能是因为里面有判断语法的问题，而语法判断和missing token是closely related.

### 3.3.3 Varying the Corruption Rate
这里还实验了不同的corruption rate, 看起来 10% - 25% 效果都差不多。

### 3.3.4 Corrupting Spans
这里实验了在固定15%的corruption rate下，span length的影响。看起来影响比较小。

### 3.3.5 Discussion
总结了一下实验的步骤。总的来说就是每次考虑一种变化，在固定最好的approach之后再继续后面的实验。
![[t5-objective.png]]
## 3.4 Pre-training Data set
这一节也比较有意思。它比较了不同pretrain dataset的影响。

### 3.4.1 Unlabeled Data Sets
这里比较了不同的dataset, 比较有意思的发现: (注意因为只pretrain了$2^{35} \approx 34B$ tokens, 所以数据集大小并不代表全train完了)

1. C4(745 GB) 比 C4 unfiltered(6.1TB)效果好。说明了data cleaning的重要性。
2. WebText-like(17GB)的效果很好。这个dataset只包含reddit上score >= 3的内容。说明data quality很重要！读好书，才能成为一个好人。
3. Wikipedia + Toronto Books Corpus效果好，单用Wikipedia效果不够好，说明domain knowledge很有用。

### 3.4.2 Pre-training Data set Size
这里讨论了数据集大小的影响。在固定$step = 2^{35} \approx 34B$ tokens的前提下，把C4 truncate到$2^{29}, 2^{27}, 2^{25}, 2^{23}$的大小，分别相当于$64,256,1024$和$4096$个epoch.

结论：随着dataset变小，模型效果下降。结合training loss曲线分析，如果重复过多次，会导致overfitting. 但是对于重复64次，模型效果下降不明显，说明少量重复(稍微多训练几个epoch)对效果影响不大。不过，对于pretraining来说，因为获取unlabeled data比较容易，多用data总是更好的。

经验规律：dataset中token的数量小于模型参数量的时候，会容易引发overfit.

## 3.5 Training Strategy
这里讨论了不同的training strategy.

### 3.5.1 Fine-tuning Methods
对于fine tuning, 讨论了几种不同的方法:

1. all parameters
2. adapter layers
3. gradual unfreezing. 这又是一个Howard提出的方法，可以去看fast.ai. 记得这种方法对image model很有效，特别是先unfreeze batch norm, 再unfreeze其他，会让收敛快很多。

最后实验的结果：

1. 跟gradual unfreezing比，其实直接fine tune所有参数效果比较好。不过gradual unfreeze会让训练更快一些。
2. adapter layers参数越多效果越好（废话）。

### 3.5.2 Multi-task Learning
在之前的实验里，都是先pretrain unsupervised再fine tune downstream. 其实也可以用multi task learning 把所有任务放在一起训练. 在这里，模型选择的时候，对**每个task**选择最好的checkpoint进行evaluate，而不是一共只选一个。 

对于multi task learning, 一个很重要的问题是怎么选择不同task data的比例。这里提到的几种方法:

1. Examples-proportioinal mixing. 也就是按照每个task dataset的大小来进行mix。因为所有的dataset里，unsupervised的大小是最大的，并且占据绝对优势，所以这里在计算mix的时候进行了一个clipping, 把dataset example的最大数量给了一个限制K。
2. Temperature-scaled mixing. 把某个task的mixing rate取一个$1/T$ power,然后normalize. 这样可以照顾到数据比较少的dataset, 让对应的task可以充分训练。当$T=1$的时候相当于Examples-proportioinal mixing, 而当$T \rightarrow \infty$的时候则相当于Equal mixing。
3. Equal mixing.

总体的结论: multitask training一般会比先pretrain再finetune差. 对不同的任务, K会有不同的最优值，而T有一个共同的最优值$T=2$.

### 3.5.3 Combining Multi-Task Learning with Fine-Tuning
其实在multi task和pretrain-finetune之外还有一种选择，就是leave one out multi task: pretrain的时候把所有其他task一起train, 然后在target task上fine tune. 

这里做了几种实验:

1. unsupervised pretrain + finetune (baseline)
2. multitask (去掉了了pretrain, 直接multitask downstream)
3. multitask pretrain + finetune
4. leave one out
5. supervised multi task pretrain

结论：
multi-task pretrain + finetune结果可以跟baseline差不多. leave one out效果没有下降很多，说明不同的task之间可以benefit。 去掉unsupervised以后效果很差，但是对translation效果影响不大，说明translation对pretraining的要求不高。

## 3.6 Scaling
这一节回答这么一个问题： 假如你有两倍的计算能力，你应该把它花在哪里？是训练更大的模型，还是训练更长的时间？还是训练几个模型来ensemble?

结论:

1. 模型大小和训练时间: $2 \times size + 2 \times trainingSteps$ 的效果约等于 $4 \times size + 1 \times trainingSteps$.
2. ensemble: $4 \times ensemble$ 不如模型扩大4倍, 但它的提升是和其他方法正交的。

## 3.7 Putting It All Together
这里介绍了最终模型使用的结构，参数和训练策略。基本上是从每个实验中取得最好的方法。然后列举了它在大量不同task上的performance。


# 4. Reflection
这一章讲了T5研究中的一些反思。

## 4.1 Takeaways
基本上又总结了一遍前面提到的各种发现。

## 4.2 Outlook
一些存在的问题和之后的方向。

1. 模型越大效果肯定越好，但也越不方便不实用，要把模型做小。
2. denoising还不够，想要更好的pretrain方法。
3. Formalizing the similarity between tasks: 使用in domain data pretrain能提高效果，但目前没有什么方法判断in domain.
4. Language-agnostic models: 发现只pretrain英语对translation task效果不够好。希望能找到(不需要multi lingual data)的更好方法。

