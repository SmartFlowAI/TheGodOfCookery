# 食神项目技术报告 V1.4

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=Mzc4ZDRjNjI1Y2NlNmFmZGRlNGZmNDNlY2JiYmZiNGVfZmttTE9NWkF0Z1NOeGhEOFB3VnhqVlRlaGlpWGlkSU9fVG9rZW46WHFJa2J0T3BHb09jVzV4TW5vS2NmNHdybjhkXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

# 0、项目基础信息

**应用DEMO地址：**

openxlab A100：12CPU 48G内存 40G显存 internlm2-chat-7b微调模型

https://openxlab.org.cn/apps/detail/zhanghui-china/shishen2024

openxlab A10：8CPU 32G内存 24G显存 internlm2-chat-1_8b 微调模型

https://openxlab.org.cn/apps/detail/zhanghui-china/shishen2024_1.8b

**模型地址：**

基于 Shanghai_AI_Laboratory/internlm2-chat-7b 微调

https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen2_full/summary

基于 Shanghai_AI_Laboratory/internlm-chat-7b 微调

https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen_full/summary

基于 Shanghai_AI_Laboratory/internlm2-chat-1_8b 微调https://www.modelscope.cn/models/zhanghuiATchina/zhangxiaobai_shishen2_1_8b/summary

**GitHub 项目链接：**

https://github.com/SmartFlowAI/TheGodOfCookery

**视频地址：**

https://www.bilibili.com/video/BV1kr421W7iA

| **章节名称** | **文档写作负责人** | **技术负责人**  |
| ------------ | ------------------ | --------------- |
| **总体概述** | 轩辕, 九月, 张辉   | 张辉            |
| **语音识别** | 轩辕               | sole fish       |
| **文生图**   | 房宇亮             | 房宇亮          |
| **RAG**      | 轩辕               | Charles，乐正萌 |
| **模型微调** | 轩辕               | 王巍龙，轩辕    |
| **Web UI**   | 房宇亮             | 房宇亮          |

# 一、总体概述

## 1.1 整体技术架构说明

​     项目主要依赖上海人工智能实验室开源模型internlm-chat-7b（包含1代和2代），在[XiaChuFang Recipe Corpus](https://opendatalab.org.cn/XiaChuFang_Recipe_Corpus) 提供的1,520,327种中国食谱数据集上借助Xtuner进行LoRA微调，形成shishen2_full模型。使用langchain将微调后的模型与chroma或faiss向量数据库整合，实现RAG检索增强的效果。并且可以进行多模态（语音、文字、图片）问答对话。前端基于streamlit实现与用户的交互。

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=MjIyOGU2NDY5OWZjNjU2ZTM3ZWVjZDk5ODU5NWJjYmRfTjRTVG5LeG5OM214enBFeW5JU2dJTXpWU042alVlOE9fVG9rZW46U3NFdmJvajI4b3FjZGp4RlRnbWNWS2thblFnXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

<div align = "center">图1 整体技术架构</div>

## 1.2 应用整体流程说明

​     用户发出请求后，应用加载各模型（语音模型，文生图模型，微调后的对话模型），并根据用户的输入类型（文字输入或者语音输入），分别进行预处理。如果未启用RAG模块，则直接调用微调后的对话模型生成回复；如果启用RAG模块，则调用langchain检索向量数据库，并将检索结果与用户输入一起输入微调后的对话模型生成回复。之后对模型回复进行格式化输出，并调用stable diffusion (SD) 模型生成图片，最后将全部结果返回用户。

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=ODEzMDlkMzYxNTI5OTE4MjlhYjIyZjJlODkwYjVkYmFfaHFweW1xTEdYUFJZZnVKRUgwTjhNUW93aDYwUjdGcGlfVG9rZW46VFdTUGJCRmNYb01kUkF4ZkNBRWM3WE10blhnXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

<div align = "center">图2 应用整体流程</div>

# 二、RAG模块

## 2.1 RAG技术概述：

RAG（Retrieval Augmented Generation, 检索增强生成）技术是一种将搜索技术和大语言模型（LLM）的提示词工程结合起来的技术。它通过从数据源中检索信息来辅助大语言模型生成答案。具体来说，RAG包括两个阶段：检索上下文相关信息和使用检索到的知识指导生成过程。在回答问题或生成文本时，先从大规模文档库中检索相关信息，然后利用这些检索到的信息来生成响应或文本，从而提高预测的质量。

根据系统的复杂性和方法的多样性，RAG可细分为朴素RAG，高级RAG和模块化RAG，其中：

- **朴素RAG**的主要步骤是索引-检索-生成，具体如下：

1. 建立索引：这一过程通常在离线状态下进行，包括数据清理、提取，将不同文件格式（如PDF、HTML、Word、Markdown等）转换为纯文本，然后进行文本分块，并创建索引。
2. 检索：使用相同的编码模型将用户输入转换为向量，计算问题嵌入和文档块嵌入之间的相似度，选择相似度最高的前K个文档块作为当前问题的增强上下文信息。
3. 生成：将给定的问题和相关文档合并为新的提示，然后由大型语言模型基于提供的信息回答问题。如果有历史对话信息，也可以合并到提示中，用于多轮对话。

- **高级RAG**旨在解决朴素RAG中存在的检索质量和生成质量的问题，主要方法是检索预处理和检索结果后处理。

1. 检索预处理：主要方法是查询重写和数据库优化。查询重写是将用户的查询送入大模型重写为多个子问题或者关键字等更详细的信息以提高检索准确度。数据库优化包括增加索引数据的细粒度、优化索引结构、添加元数据等策略。
2. 检索结果后处理**：**主要通过Reranker模型对检索结果进行重排以进一步筛选最相关的结果。或者对检索结果进行压缩以突出关键信息，同时一定程度上加快推理速度。

- **模块化RAG**是一种更灵活的RAG实现，它允许不同的检索和生成模块根据特定的应用需求进行替换或重新配置，为整个问答过程提供了更丰富的多样性和更强的灵活性。模块化RAG范式正成为RAG领域的主流。

目前较为成熟的集成了RAG实现的开发框架有 [LangChain](https://www.langchain.com/) 和 [llama-index](https://www.llamaindex.ai/)等。本项目主要基于LangChain搭建RAG系统，并探索优化RAG性能的各种方法。

## 2.2 基于Langchain的RAG技术简述

LangChain 是一个开源框架，用于构建基于大型语言模型（LLM）的应用程序。LLM 是基于大量数据预先训练的大型深度学习模型，可以生成对用户查询的响应，例如回答问题或根据基于文本的提示创建图像。LangChain 提供各种工具和抽象，以提高模型生成的信息的定制性、准确性和相关性。例如，开发人员可以使用 LangChain 组件来构建新的提示链或自定义现有模板。LangChain 还包括一些组件，可让 LLM 无需重新训练即可访问新的数据集。LangChain 使得应用程序能够：

- **具有上下文感知能力**：将语言模型连接到上下文来源（提示指令，少量的示例，需要回应的内容等）
- **具有推理能力**：依赖语言模型进行推理（根据提供的上下文如何回答，采取什么行动等）

LangChain 框架由以下部分组成。

- **[LangChain 库](https://python.langchain.com.cn/docs/)**：Python 和 JavaScript 库。包含了各种组件的接口和集成，一个基本的运行时，用于将这些组件组合成链和代理，以及现成的链和代理的实现。
- **[LangChain 模板](https://python.langchain.com.cn/docs/templates)**：一系列易于部署的参考架构，用于各种任务。
- **[LangServe](https://python.langchain.com.cn/docs/langserve)**：一个用于将 LangChain 链部署为 REST API 的库。
- **[LangSmith](https://python.langchain.com.cn/docs/langsmith)**：一个开发者平台，让你可以调试、测试、评估和监控基于任何 LLM 框架构建的链，并且与 LangChain 无缝集成。

下图展示了 LangChain 框架的层次组织结构，显示了跨多个层次的部分之间的相互连接。

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDM5OGE0NzkzMTAzNjJjNTVmZWNhZWY4Y2E3NWViNDdfSElxZkl5bDJBV3A3anJzYkg0eHc4UEdJYVhzRkRhR21fVG9rZW46QTZjVWJBek55b3EwaFF4YzU5M2NvUVgybldnXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

<div align = "center">图 3 LangChain 框架层次组织结构</div>

这些部分一起简化了整个应用程序的生命周期：

- 开发：在 LangChain/LangChain.js 中编写应用程序。使用模板作为参考，快速开始。
- 生产化：使用 LangSmith 来检查、测试和监控链，这样可以不断改进并有信心地部署。
- 部署：使用 LangServe 将任何链转换为 API。

## 2.3 基于chroma+sqlite的RAG技术说明

[Chroma](https://www.trychroma.com/)是一款AI 原生的开源轻量级向量数据库，可以存储文档、图片等数据的向量及其元数据，并进行搜索，其最大的优势是简单易用。它目前只支持CPU计算，但可以利用乘积量化的方法，将一个向量的维度切成多段，每段分别进行k-means，从而减少存储空间和提高检索效率。LangChain中集成了Chroma相关的API，使得开发者可以快捷地构建基于LLM的应用。

[SQLite](https://www.sqlite.org/) 是一个开源的嵌入式关系数据库，实现了自给自足的、无服务器的、配置无需的、事务性的 SQL 数据库引擎。它是一个零配置的数据库，这意味着与其他数据库系统，比如 MySQL、PostgreSQL 等不同，SQLite 不需要在系统中设置和管理一个单独的服务。这也使得 SQLite 是一种非常轻量级的数据库解决方案，非常适合小型项目、嵌入式数据库或者测试环境中。SQLite 的一些主要特性包括：

1. 无服务器的：SQLite 不是一个单独的服务进程，而是直接嵌入到应用程序中。它直接读取和写入磁盘文件。
2. 事务性的：SQLite 支持 ACID（原子性、一致性、隔离性、持久性）属性，能够确保所有事务都是安全、一致的，即使在系统崩溃或者电力中断的情况下。
3. 零配置的：SQLite 不需要任何配置或者管理，这使得它非常容易安装和使用。
4. 自包含的：SQLite 是一个自包含系统，这意味着它几乎不依赖其他任何外部系统或者库，这使得 SQLite 的跨平台移植非常方便。
5. 小型的：SQLite 非常小巧轻量，全功能的 SQLite 数据库引擎的大小只有几百KB。
6. 广泛应用：SQLite 被广泛应用在各种各样的产品和系统中，包括手机、平板电脑、嵌入式系统、物联网设备等。它也被广泛用于网站开发、科学研究、数据分析等领域。

在一些轻量级的应用场景下，SQLite 是一个非常理想的选择，因为它简单、高效、易于使用和部署。然而，对于需要处理大量并发写操作或者需要更高级的功能（如用户管理或者存储过程等）的应用场景，更全功能的数据库系统（如 PostgreSQL 或 MySQL）可能会是更好的选择。

## 2.4 基于faiss的RAG技术说明

[Faiss](https://faiss.ai/index.html)的全称是Facebook AI Similarity Search，是Facebook（Meta）开源的一款高性能向量数据库。与Chroma相比，支持更多高性能向量检索与量化压缩算法，并且支持GPU运算，可以处理十亿级数据集。

LangChain中同样集成了Faiss相关的API，可以简单快捷地使用Faiss构建文本向量数据库，尤其适合RAG数据集规模较大的情况。

## 2.5 基于BCEembedding的二阶段检索算法说明

[BCEmbedding](https://github.com/netease-youdao/BCEmbedding/blob/master/README_zh.md)（Bilingual and Crosslingual Embedding）是网易有道研发的两阶段检索算法库，作为网易有道开源RAG项目QAnything的基石发挥着重要作用。

BCEmbedding以其出色的双语和跨语种能力而著称，在语义检索中消除中英语言之间的差异，从而实现：

- **强大的双语和跨语种语义表征能力【****[基于MTEB的语义表征评测指标](https://github.com/netease-youdao/BCEmbedding/blob/master/README_zh.md#基于mteb的语义表征评测指标)****】。**
- **基于LlamaIndex的RAG评测，表现SOTA【****[基于LlamaIndex的RAG评测指标](https://github.com/netease-youdao/BCEmbedding/blob/master/README_zh.md#基于llamaindex的rag评测指标)****】。**

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=OWU4MzdlNGNkYTA5Y2EyODQxYmM2NzdiZTEyM2QxZmFfanIzamxPcjcyWmprbTZBcVpWNlFLY3NEQm9FbDBSV2hfVG9rZW46RlJCZ2JFd3Zib0NhRDJ4dE5uN2NDSm1IbmxnXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

<div align = "center">图4 BCEmbedding评测表现</div>

BCEmbedding是一个二阶段检索算法模型库，由召回和精排这两个模块组成，核心是EmbeddingModel(bce-embedding-base_v1)和RerankerModel(bce-reranker-base_v1)两款模型。

bce-embedding-base_v1模型是一款基于XLMRoberta的句子向量编码模型，拥有中英双语种编码能力。用于离线提取知识库语料提取语义向量，并由向量数据库保存。用户提问时，该模型实时地提取用户问题的语义向量，用于之后结合向量数据库进行一阶段检索。

bce-reranker-base_v1模型同样是一款基于XLMRoberta的句子向量编码模型，拥有中英日韩四个语种跨语种语义精排能力。一阶段检索后，该模型依次评估各文档与用户问题的相关性，输出有意义的语义相关分数。之后根据相关分数进行过滤和排序，实现对文档的选优。

二阶段检索算法结合召回和精排二者的优势，召回阶段可以快速找到用户问题相关文本片段，精排阶段可以将正确相关片段尽可能排在靠前位置，并过滤掉低质量的片段。二阶段检索可以很好地权衡检索效果和效率，具有巨大应用价值。

## 2.6 本项目RAG方案--假设问题编码（HyQE）详解

在搭建RAG模块的第一阶段，我们采用最常用的文本块编码方案，将整个txt格式的菜谱文件按标点符号和段落长度分块，对分块的句向量进行编码和检索。实验发现该方案表现很差，检索器返回的都是零碎的菜谱片段，或者是包含多个菜谱片段的混合片段。针对这一问题，我们放弃了常用的文本分块器，将每个的菜谱作为一个整体进行编码，确保检索得到的都是完整菜谱。后期实践中发现，即使采用了BCEembedding二阶段检索算法，我们的RAG模块的召回率仍然很低，检索器返回的文档大部分都是相似但不准确符合用户提问的菜谱。之后加入了更关注关键字词的BM25检索器，与基于向量的检索器组成组合式检索器，依旧不能解决该问题。

在深入的交流和思考后，我们认为，项目中用户的输入一般是“xx菜怎么做”之类的问题，而被检索的对象，则是长度几百字的菜谱。目前方案下的RAG模块，最根本的任务相当于是要实现【问题&答案】的相似度精确匹配与检索，而“xx菜怎么做”之类的问题和菜谱的句子内容本身区别就极大，即使EmbeddingModel可以精确编码，检索时出现混淆也几乎是必然。

在寻找解决方案的过程中，两款的高级RAG方案给予了我们启发。第一个是假设文档编码（Hypothetical Document Embeddings，HyDE）。在采用了该方案的RAG系统中，用户输入的问题，会先交给LLM生成一个假设性回答，再根据假设回答的句向量在数据库中搜索，得到准确的参考文档。这种【答案&答案】的检索方案能够有效地弥补【问题&答案】之间的语义差距。另一个参考方案是开源知识库FastGPT采用的问答对编码策略。FastGPT系统会在导入数据后，使用LLM为每段文本生成一个假设问题，形成问答对，通过【问题&问答对】的综合检索，实现对用户问题相关的文档的精确召回。

结合本项目实际情况考虑，我们微调使用的数据集格式为{"input": "xxx菜怎么做", "output": "完整的菜谱"}。显然，我们可以将input部分作为编码与检索的对象，与用户输入的“xx菜怎么做”类型的问题进行相似度匹配，实现【问题&问题】检索。在langchain中实现该方案，只需要将output部分的菜谱保存在content为input部分的Document对象的metadata中。在检索器完成检索后，将结果文档输入大模型前，从Document对象的metadata中取出菜谱，并用其替换掉该对象的content部分的内容即可。

​      同时考虑到本项目完整数据集有150万份菜谱，我们最终采用了高性能的Faiss作为RAG模块的默认向量数据库。但也保留了对轻量级的Chroma向量数据库的支持。两个向量数据库的互相切换可以在项目启动前通过修改配置文件实现。

采用HyQE方案后，本项目RAG系统达到了近乎100%的召回率。同时，编码和保存短问题的资源消耗也远少于长菜谱。但是，该方案也存在两个明显缺陷：第一，仅适用于拥有现成的一问一答格式的数据集的项目，如果是常见的连续长文档格式的RAG数据集，就要考虑合适的文档分块与假设问题生成策略，建议参考FastGPT项目的方案。第二，用户输入仅限于“xx菜怎么做”类型的问题这一假设太强，尤其不符合多轮对话场景的后续问题。解决该问题可以考虑使用基于大模型的上下文压缩器，将对话历史和用户当前问题重写为一个新的问题。langchain框架在多轮对话RAG链的实现中就采用了这一方案。但是压缩过程中的信息损失比较严重，生成的新问题质量也难以保证。更合适的解决方法，我们也仍在探索当中。

## 2.7 参考

1. Retrieval-Augmented Generation for Large Language Models: A Survey  https://arxiv.org/abs/2312.10997 
2. langchain中文网：https://python.langchain.com.cn/docs/get_started/introduction
3. langchain官方文档：https://python.langchain.com/docs/get_started/introduction
4. chroma官网：https://www.trychroma.com/
5. chroma仓库：https://github.com/chroma-core/chroma
6. faiss文档：https://faiss.ai/index.html
7. faiss仓库：https://github.com/facebookresearch/faiss
8. 为RAG而生-BCE embedding技术报告：https://zhuanlan.zhihu.com/p/681370855
9. BCEmbedding代码仓库：
10. [Improving retrieval using Hypothetical Document Embeddings(HyDE)](https://medium.aiplanet.com/advanced-rag-improving-retrieval-using-hypothetical-document-embeddings-hyde-1421a8ec075a)
11. FastGPT文档：https://doc.fastgpt.in/docs/

# 三、文生图模块

## 3.1 文生图概述：

文本到图像生成模型一般以自然语言描述为输入，输出与该描述相匹配的图像，现已被广泛应用于艺术创作、游戏设计、虚拟现实等领域。文生图技术的发展推动了图像生成的自动化和个性化，使得创造新图像变得更加快捷和多样化。文生图技术不仅仅限于生成对抗网络（GANs），还包括多种不同的方法和模型，如变分自编码器（VAEs）、自回归模型、扩散模型等。

- 生成对抗网络（GANs）

生成对抗网络是最著名的文生图技术之一，由两个神经网络组成：生成器和鉴别器。生成器的目标是创造出足够真实的图像，以至于鉴别器无法区分它和真实图像的区别。而鉴别器的目标则是正确区分出哪些是真实图像，哪些是由生成器创造的。通过这种对抗过程，生成器学习如何产生越来越真实的图像。GANs已被用于各种应用，从艺术创作到数据增强等。

- 变分自编码器（VAEs）

变分自编码器是另一种流行的图像生成方法。VAEs通过学习输入数据的潜在表示来生成新图像。它们由两部分组成：编码器和解码器。编码器将输入数据压缩成一个潜在空间中的表示，而解码器则将这个表示转换回原始数据。VAEs通常用于生成模糊或梦幻般的图像，适合于风格转换和图像编辑等任务。

- 自回归模型

自回归模型是基于序列生成的方法，每次生成图像的一部分，然后将其作为下一步的输入。这种模型包括像PixelRNN和PixelCNN这样的网络，能够生成高质量的图像，但生成过程通常比GANs和VAEs更慢。

- 扩散模型

近年来，扩散模型成为了生成图像技术的新星，它们通过逐渐从一个随机噪声分布中删除噪声来生成图像。这一过程与物理过程中的扩散相似，因此得名。扩散模型已显示出在生成高质量图像方面的巨大潜力，尤其是在细节和多样性方面。

## 3.2 SD文生图技术简介

StableDiffusion(SD)是CompVis、Stability AI和LAION等公司研发的一个开源文生图模型，训练数据为[LAION-5B](https://link.zhihu.com/?target=https%3A//laion.ai/blog/laion-5b/)（同样开源）。SD是一个基于latent的扩散模型，它在UNet中引入text condition来实现基于文本生成图像。SD的核心来源[Latent Diffusion](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2112.10752)这个工作，常规的扩散模型是基于pixel的生成模型，而Latent Diffusion是基于latent的生成模型，它先采用一个autoencoder将图像压缩到latent空间，然后用扩散模型来生成图像的latents，最后送入autoencoder的decoder模块就可以得到生成的图像。

SD模型的主体结构如下图所示，主要包括三个模型：

- autoencoder：encoder将图像压缩到latent空间，而decoder将latent解码为图像；
- CLIP text encoder：提取输入text的text embeddings，通过cross attention方式送入扩散模型的UNet中作为condition；
- UNet：扩散模型的主体，用来实现文本引导下的latent生成。

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=NWIzZWU2NzJhYmE2MDM2ZmNlYzY0MDNlODk1ZTg3MjNfd0tmc2NBRVp0U1JxQWNDZUFicUhzeUU1M0dvZXlON1hfVG9rZW46S2VkN2IwMHYxb3cwclF4ak42bGNHZHc5bjV1XzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

<div align = "center">图5 StableDiffusion模型架构</div>

## 3.3 智谱API文生图技术简介

​    智谱AI以文生图CogView大模型能够根据简短的中英文文字描述生成一张图片，其背后有强大的跨模态预训练大模型技术支持，该模型采用Transformer+VQVAE架构，能够增强跨模态对大模型的理解和创新，在预训练过程能够同时学习模态间和模态内的多种关联性，提升“图像”和“文本”跨模态语义匹配效果，将“文生成图”和“图生成文”任务融合到同一个模型进行端到端学习，从而增强文本和图像的跨模态语义对齐。

​     该模型面向的用户人群非常广泛，可以帮助自媒体编辑生成文章配图，为设计师提供创意参考和素材来源。它既能启发画师、设计师、艺术家等专业视觉内容创作者的灵感，辅助其进行艺术创作，还能为媒体、作者等文字内容创作者提供高质量、高效率的配图，而且可以以较低的成本给出用户所需要的画面的图片，图片不存在版权及肖像权问题，且支持多种图片风格。目前已支持国风、油画、水彩、水粉、动漫、写实等八种不同风格高清画作的生成，还支持六种主题（动物、人物、风景、建筑、食物、其他）的图像生成。     

​    技术环节，得益于在跨模态预训练大模型上的技术创新，其性能在MS COCO上超过Open AI 的DALL.E，实现超分辨率生成。 

## 3.4 文生图技术选型以及在本项目的实践

本项目使用的文生图模型的主要原则是：

1. 模型免费
2. 生成图片速度在用户可接受到的时间范围

出于以上原则的考虑，我们放弃以智谱文生图作为本项目文生图技术的实现（收费，且可能存在版权问题），而采用以开源StableDiffusion基础模型。目前项目组找到的StableDiffusion比较流行的基础版本有以下几个：

- [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)，由Robin Rombach, Patrick Esser开发，是一个 [Latent Diffusion Model](https://arxiv.org/abs/2112.10752)，使用一个固定的，预先训练的文本编码器(CLIP ViT-L/14)。
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)，由Stability AI开发，是一个 [Latent Diffusion Model](https://arxiv.org/abs/2112.10752)，使用两个固定的，预先训练的文本编码器(OpenCLIP-ViT/G 和 CLIP-ViT/L)。
- [Taiyi-Stable-Diffusion-1B-Chinese-v0.1](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1)，由IDEA（粤港澳大湾区数字经济研究院）开发，是首个开源的中文Stable Diffusion模型，基于0.2亿筛选过的中文图文对训练。

通过分析，要在本项目里使用文生图功能，需要满足以下几点要求：

1. 部署模型显存大小适合
2. 生图内容准确，和提示词匹配
3. 生图速度快，等待时间短

考虑到目前食神应用综合了语音识别、文生图等能力，为了让应用能够再更小的显存配置中顺利运行，项目放弃选择占用显存较大的[stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)模型，而是在开始时尝试用[stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) 来生图。但是因为该模型输入语言为英文，而本项目的使用语言为中文，所以其生图结果准确性无法满足需求。之后，项目组在huggingface上查找可以直接接受中文输入的SD模型，最终确定以Taiyi-Stable-Diffusion-1B-Chinese-v0.1模型作为本项目文生图模型。

经过验证，使用Taiyi-Stable-Diffusion-1B-Chinese-v0.1模型生成的菜谱更加接近于中国人的理解水平。该模型生成1张图片的时间约2s左右，可以基本满足时延要求，同时测试验证表明，该模型显存占用只需2G左右。

## 3.5 参考

1. [维基百科文生图](https://zh.wikipedia.org/wiki/文本到图像生成模型)
2. 文生图模型之Stable Diffusion：https://zhuanlan.zhihu.com/p/617134893
3. 智谱介绍：https://www.paratera.com/cpfw/ay/detail/373.html
4. DDPM的奠基之作：[《Denoising Diffusion Probabilistic Models》](https://arxiv.org/abs/2006.11239)
5. Diffusion Models扩散模型与深度学习(数学原理和代码解读)：https://xduwq.blog.csdn.net/article/details/118724666
6. Hugging Face 扩散模型课程**：**https://huggingface.co/datasets/HuggingFace-CN-community/Diffusion-book-cn

# 四、语音识别模块

## 4.1 语音识别概述

​     语音识别，即让机器“听懂”人类的语言。技术上，通常将语音识别称为自动语音识别（Automatic Speech Recognition，ASR），主要是将人类语音中的词汇内容转换为计算机可读的输入，一般都是文本内容，也有可能是二进制编码或者字符序列。狭义上，语音识别一般专指语音转文字的过程，简称语音转文本识别[（Speech To Text, STT）](https://www.audiology.org/news-and-publications/audiology-today/articles/communication-strategies-and-speech-to-text-transcription/)，与语音合成[(Text To Speech, TTS)](https://www.naturalreaders.com/online/)对应。

## 4.2 whisper语音技术简介

​     Whisper是OpenAI于2022年12月发布的开源语音处理系统，它使用了从网络上收集的68万个小时的多语言和多任务监督数据进行训练，从而在英语语音识别方面达到了接近人类水平的鲁棒性和准确性。尽管论文名为[Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)，但Whisper不只是具有语音识别能力，还具备[语音活性检测(Voice activity detection，VAD)](https://zh.wikipedia.org/wiki/语音活性检测)、声纹识别（Voiceprint Recognition）、语音翻译（其他语种语音到英语的翻译）等能力。Whisper是一种端到端的语音系统，相比于之前的端到端语音识别，其特点主要是：

- 多语种：英语为主，支持99种语言，包括中文。
- 多任务：语音识别为主，支持VAD、语种识别、说话人日志、语音翻译、对齐等。
- 数据量：68万小时语音数据用于训练，从公开数据集或者网络上获取的多种语言语音数据，远超之前语音识别几百、几千、最多1万小时的数据量。
- 鲁棒性：主要还是源于海量的训练数据，并在语音数据上进行了常见的增强操作，例如变速、加噪、谱增强等。
- 多模型：提供了从tiny到large，从小到大的五种规格模型，适合不同场景。如下图所示：

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=NDRiYThlNjI5ZmUyZjlkNjY4YjVmZTZjNThjMjRjMjZfcVR0M25ZZlF2ZXZyWVpqbUN5OTdYb010YWNzVDRrR21fVG9rZW46SlA4YWJPT1lwb1l2VVB4ekExMGNjMmJTbnNmXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

图3 Whisper多规格模型

​     如下图所示，Whisper的架构是一个简单的端到端（end to end）方法，采用了编码器-解码器的[Transformer](https://arxiv.org/abs/1706.03762)模型实现。在工作过程中，输入的音频首先被分成30秒的块，并转换成log-Mel频谱图，然后传递到编码器。解码器则负责预测相应的文本标题，并与特殊token标记混合，这些标记指导单个模型执行诸如语言识别（language identification）、短语级时间戳（phrase-level timestamps）、多语言语音转录（multilingual speech transcription）和英语语音翻译（to-English speech translation）等任务。

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=MGY0M2EyMTFlOGMxNDg0ZTdjYjQzZGVjY2IyZDdjZTRfNnppRzZkWGxqNnZObFJDYXJJbXlmU2lkTzJPSlJLeUJfVG9rZW46RWJORGJobmQ0b0p1Wkl4bVBXY2NrRnlqbmtkXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

<div align = "center">图6 Whisper模型架构</div>

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=NmU3ZDUzNWE2N2VjMjJiMDZjZjRiMjcxNTQwZWMyNWVfNGRmOUt2YzlZSGtob1F1MDJFemg0c1doRHZ4TUtQd1dfVG9rZW46UnAwRGJCemwybzh1clB4czdFUWNyVFJLbmNoXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

<div align = "center">图7 Whisper训练过程</div>

## 4.3 paraformer语音技术简介

​     **Paraformer是一种全新的语音识别模型**，由阿里达摩院发布。该模型是业界首个应用落地的非自回归端到端语音识别模型，它在推理效率上最高可较传统模型提升10倍，并且在多个权威数据集上的识别准确率名列第一。    Paraformer语音技术主要解决了端到端识别效果与效率兼顾的难题。它采用了单轮非自回归模型的设计，通过创新的预测器设计，实现对目标文字个数及对应声学隐变量的高准确度预测。此外，Paraformer还引入了机器翻译领域的浏览语言模型思路，显著增强了模型对上下文语义的建模能力。在训练过程中，Paraformer使用了长达数万小时、覆盖丰富场景的超大规模工业数据集，进一步提升了识别准确率。

​     Paraformer模型结构如下图所示，由 Encoder、Predictor、Sampler、Decoder 与 Loss function 五部分组成。其核心点主要有：

- Predictor 模块：基于 CIF 的 Predictor 来预测语音中目标文字个数以及抽取目标文字对应的声学特征向量
- Sampler：通过采样，将声学特征向量与目标文字向量变换成含有语义信息的特征向量，配合双向的 Decoder 来增强模型对于上下文的建模能力
- 基于负样本采样的 MWER 训练准则

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=ZmZjODhjM2VkYmI0YWQ3NWQ1ZDg3N2JhZmVjMDM3YTBfMmpMSHI0QTVhNWVQa2NpYTJvZlM2UmdRWjZHNDhoOEdfVG9rZW46REI0RGJoWDNpb0VUV1d4d2k0eWNLakRwbkJkXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

<div align = "center">图8 Paraformer模型结构</div>

​     Paraformer语音技术的应用场景十分广泛，包括语音输入法、智能客服、车载导航、会议纪要等。无论是需要快速准确地识别语音输入的场景，还是需要处理复杂语音交互的场合，Paraformer都能提供出色的性能表现。具体而言，Paraformer可应用于：

- 对语音识别结果返回的即时性有严格要求的实时场景，如实时会议记录、实时直播字幕、电话客服等。
- 对音视频文件中语音内容的识别，从而进行内容理解分析、字幕生成等。
- 对电话客服呼叫中心录音进行识别，从而进行客服质检等。

## 4.4 语音识别技术选型以及在本项目的实践

​    出于项目对语音识别即时性，以及对于中文菜名识别效果的考量，本项目最终采用了阿里开源的paraformer语音识别模型进行识别。 

其实，本项目在最开始的时候尝试采用OpenAI的whisper 模型作为语音识别模型，但是经过实验之后发现whisper模型对于中文菜名的识别效果并不好。只有medium以上模型才能够较为准确的对中文菜名进行识别，因此在食神的最初版采用了whisper的medium模型进行语音识别。但由于medium大小的whisper模型所需算力比较高，在线部署的模型进行语音识别需要的时间比较长，因此初版语音识别模型的识别延迟较高。

​     随后项目组开始寻找一些开源的，同时对中文支持比较好的轻量化的语音识别模型，最终选择了paraformer语音识别模型，paraformer模型相对于whisper，对中文的支持更好，同时也更加轻量化。在将paraformer迁移到食神项目的过程中，遇到的主要问题在于paraformer的版本和modelscope库的版本冲突问题。为了避免更改modelscope的版本依赖，项目组选择了采用funasr库对paraformer进行模型调用，而没有采用modelscope自带的api。funasr是达摩院的一个语音识别库，但是其不支持本地模型调用，为此项目组重新实现了funasr库的模型加载过程，支持调用本地模型。

## 4.5 参考

1. 语音识别技术简史：https://zhuanlan.zhihu.com/p/82872145
2. 语音识别研究综述：http://www.c-s-a.org.cn/html/2022/1/8323.htm
3. 官网Whisper入口：https://openai.com/research/whisper
4. Whisper论文：[Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)
5. Whisper代码仓库：https://github.com/openai/whisper
6. Whisper的model card：https://github.com/openai/whisper/blob/main/model-card.md
7. Whisper介绍：https://zhuanlan.zhihu.com/p/662906303
8. 阿里官方paraformer介绍：https://help.aliyun.com/zh/dashscope/developer-reference/quick-start-7?spm=a2c4g.11186623.0.i2
9. paraformer论文：[Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317)
10. paraformer论文解读：[Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型](https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw)
11. 必读文献综述：[End-to-End Speech Recognition: A Survey](https://arxiv.org/abs/2303.03329)
12. 台大李宏毅语音识别课程：https://www.bilibili.com/video/BV1cU4y1U7gi/

# 五、模型微调

## 5.1 模型微调概述

​       大模型微调是一种迁移学习技术，通过在预训练模型的基础上进行额外训练，使其适应特定任务或领域。这一过程包括选择预训练模型，准备目标任务的数据，调整模型结构，进行微调训练，以及评估和部署。微调的优点在于节省时间和资源，提高性能，但也存在过拟合风险和模型选择与调整的复杂性。总体而言，它是一种强大的技术，特别适用于数据受限或计算资源有限的情况。在 OpenAI 发布的 ChatGPT 中，就主要应用了大模型微调技术，从而获得了惊艳全世界的效果。

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=MDU5YjJhMWJmMDBmYWZhOTE0ZDJiNzQ3MDdkYzkwZDFfVkI0a0V6U21WeTJ6SVVqZFFmekJNTE9GVzNOZlRsQUJfVG9rZW46U0VoZGJPa1RJb2ZhV3B4SWpvd2N5Q1ZObkZlXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

<div align = "center">图9 ChatGPT模型微调</div>

如下图所示，微调由以下4步构成。

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=ODlhNWM3NzgzNDM0NmUxMmM1OGE0ODE1ZTQ1NTg1OTBfbXVTeE40b1NjMUQ5YXg2V0J2MVlzODNTTFlTYlFWazFfVG9rZW46QTg5N2JOODV5b2lXekp4cTNNMGM4Y3V3bjRmXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

<div align = "center">图10  模型微调步骤</div>

1. 在源数据集上预训练一个神经网络模型，即源模型。
2. 创建一个新的神经网络模型，即目标模型。它复制了源模型上除了输出层外的所有模型设计及其参数。我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。我们还假设源模型的输出层与源数据集的标签紧密相关，因此在目标模型中不予采用。
3. 为目标模型添加一个输出大小为目标数据集类别个数的输出层，并随机初始化该层的模型参数。
4. 在目标数据集上训练目标模型。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。

当目标数据集远小于源数据集时，微调有助于提升模型的泛化能力。

## 5.2  基于Xtuner的模型微调技术简介

XTuner 是一个高效、灵活、全能的轻量化大模型微调工具库。

**高效**

- 支持大语言模型 LLM、多模态图文模型 VLM 的预训练及轻量级微调。XTuner 支持在 8GB 显存下微调 7B 模型，同时也支持多节点跨设备微调更大尺度模型（70B+）。
- 自动分发高性能算子（如 FlashAttention、Triton kernels 等）以加速训练吞吐。
- 兼容 [DeepSpeed](https://github.com/microsoft/DeepSpeed) ，轻松应用各种 ZeRO 训练优化策略。

**灵活**

- 支持多种大语言模型，包括但不限于 [InternLM](https://huggingface.co/internlm)、[Mixtral-8x7B](https://huggingface.co/mistralai)、[Llama2](https://huggingface.co/meta-llama)、[ChatGLM](https://huggingface.co/THUDM)、[Qwen](https://huggingface.co/Qwen)、[Baichuan](https://huggingface.co/baichuan-inc)。
- 支持多模态图文模型 LLaVA 的预训练与微调。利用 XTuner 训得模型 [LLaVA-InternLM2-20B](https://huggingface.co/xtuner/llava-internlm2-20b) 表现优异。
- 精心设计的数据管道，兼容任意数据格式，开源数据或自定义数据皆可快速上手。
- 支持 [QLoRA](http://arxiv.org/abs/2305.14314)、[LoRA](http://arxiv.org/abs/2106.09685)、全量参数微调等多种微调算法，支撑用户根据具体需求作出最优选择。

**全能**

- 支持增量预训练、指令微调与 Agent 微调。
- 预定义众多开源对话模版，支持与开源或训练所得模型进行对话。
- 训练所得模型可无缝接入部署工具库 [LMDeploy](https://github.com/InternLM/lmdeploy)、大规模评测工具库 [OpenCompass](https://github.com/open-compass/opencompass) 及 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)。

## 5.3 模型微调的数据集准备

本文在XiaChuFang Recipe Corpus 提供的1,520,327种中国食谱数据集进行微调。数据集链接如下：https://opendatalab.org.cn/XiaChuFang_Recipe_Corpus

从openxlab下载的原始数据集：XiaChuFang_Recipe_Corpus.tar.gz 解压后每一行如下所示：

```Plain
{"name": "西班牙金枪鱼沙拉", "dish": "金枪鱼沙拉", "description": "", "recipeIngredient": ["超市罐头装半盒金枪鱼(in spring water)", "2大片生菜", "5个圣女果", "半根黄瓜", "半个红柿椒", "半个紫洋葱", "1个七成熟水煮蛋", "适量红酒醋", "适量胡椒", "适量橄榄油"], "recipeInstructions": ["鸡蛋进水煮，七成熟捞出（依个人喜好），同时备其他菜", "生菜撕片，圣女果开半，黄瓜滚刀，红柿椒切丝，紫洋葱切丝，鸡蛋四均分", "金枪鱼去水", "撒黑胡椒，红酒醋和少许橄榄油", "拌匀，拍照，开动"], "author": "author_67696", "keywords": ["西班牙金枪鱼沙拉的做法", "西班牙金枪鱼沙拉的家常做法", "西班牙金枪鱼沙拉的详细做法", "西班牙金枪鱼沙拉怎么做", "西班牙金枪鱼沙拉的最正宗做法", "沙拉"]}
```

利用文心一言进行辅助编程，将该格式的数据集转换为xtuner需要的格式：

```Plain
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```

生成的微调数据集如下所示：

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=MTQ1MGI1OGE5NDZiMGU3OWIyZmYyODI2YjFkZjYxNjNfa0V4ZG9iVEhHNXUwTUdJaGRrZ1B3NTBjS2g5azdSRzVfVG9rZW46RjRBSmI2S0Nab1hCWFd4a2JTcmNWdVpIbkhiXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

<div align = "center">图11 微调数据集</div>

## 5.4 微调方法及其产出物

使用xtuner对菜谱的微调数据集进行微调，采用以下的通用步骤：

1. 根据config微调模型：
   1. ```Plain
      xtuner train ${YOUR_CONFIG} --deepspeed deepspeed_zero2
      ```

   2. `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 来优化训练过程。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 等。
2. 将保存的 `.pth` 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 LoRA 模型：

```Bash
xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH_file_dir} ${SAVE_PATH}
```

1. 3.将LoRA模型合并入 HuggingFace 模型：

```Plain
xtuner convert merge ${Base_PATH} ${LoRA_PATH} ${SAVE_PATH}
```

1. 对于生成好的 HuggingFace 模型，可以使用以下命令对其进行验证：
   1. ```Plain
      xtuner chat ${SAVE_PATH} [optional arguments]
      ```

参数：

- `--prompt-template`: 一代模型使用 internlm_chat，二代使用 internlm2_chat。
- `--system`: 指定对话的系统字段。
- `--bits {4,8,None}`: 指定 LLM 的比特数。默认为 fp16。
- `--no-streamer`: 是否移除 streamer。
- `--top`: 对于二代模型，建议为0.8。
- `--temperature`: 对于二代模型，建议为0.8。
- `--repetition-penalty`: 对于二代7b模型，建议为1.005，对于一代模型可不填。
- 更多信息，请执行 `xtuner chat -h` 查看。

## 5.5 微调模型的使用

经过微调训练后的模型，不仅可以采用通用的方式加载，还支持使用BitsAndBytes方式进行4bit量化加载。

正常加载：

```Plain
model = (
            AutoModelForCausalLM.from_pretrained(llm_model_path, trust_remote_code=True)
            .to(torch.bfloat16)
            .cuda()
        )
tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
```

4bit量化加载：

```Python
 quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
 model = AutoModelForCausalLM.from_pretrained(llm_model_path, trust_remote_code=True, torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config).eval()
 tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
```

使用BitsAndBytes进行4bit加载方式，**无需提前将原始模型转换为4bit模型**，即可完成加载步骤。

## 5.6 参考

1. 大模型微调总结：https://www.zhihu.com/tardis/zm/art/627642632?source_id=1003
2. 动手学深度学习：http://zh.gluon.ai/chapter_computer-vision/fine-tuning.html
3. xtuner官方文档：https://github.com/InternLM/xtuner/blob/main/README_zh-CN.md
4. 微调数据集下载：[150万下厨房微调数据集:提取密码8489](https://pan.baidu.com/s/1TyqDWRI5jOs621VXr-uMoQ)

# 六、应用Web UI技术

## 6.1 Web UI概述

尽管常见的Web UI技术包含：JavaScript、jQuery、Vue.js 、React.js、Angular.js等，但是在数据应用领域，特别是人工智能领域，Streamlit和Gradio已经成为了构建数据应用程序WebUI的主流框架。

**Streamlit** 是一种用于构建数据应用程序的 Python 库：

- Streamlit 主要用于构建数据科学和机器学习应用程序，通过简单的 Python 脚本即可实现交互式的数据可视化和分析。
- Streamlit 提供了快速迭代、简单易用的特性，使得数据科学家和分析师可以更快地创建漂亮的数据应用程序，而无需深入学习前端开发技术。
- 与 JavaScript 技术相比，Streamlit 在构建数据应用程序方面更为方便，但在灵活性和定制化方面可能会有所不足。

**Gradio** 是一个用于快速构建机器学习界面的 Python 库。Gradio 也旨在简化机器学习模型的部署和展示，但它专注于提供一个简单易用的界面来展示模型的输入和输出，而不涉及其他数据分析或可视化方面的功能。

Gradio 和 Streamlit 之间的主要区别包括：

1. 重点功能：
   1. Gradio 主要用于构建机器学习模型的交互式界面，使用户能够直观地了解模型的输入和输出，并与模型进行交互。
   2. Streamlit 则更加通用，可以用于构建各种类型的数据应用程序，包括数据分析、可视化、文本处理等，不仅局限于机器学习模型的展示。
2. 界面设计：
   1. Gradio 提供了一系列预定义的界面组件，如输入框、滑块、下拉框等，使用户可以轻松地构建简单且美观的界面。
   2. Streamlit 也提供了类似的界面组件，但其更加灵活，用户可以通过代码自定义界面的外观和交互方式，从而实现更多样化的应用程序设计。
3. 学习曲线：
   1. Gradio 的学习曲线相对较低，适合那些想要快速构建机器学习模型展示界面的用户，尤其是对前端开发不太熟悉的人员。
   2. Streamlit 的学习曲线可能稍高一些，但提供了更多的灵活性和定制化选项，适用于更广泛范围的数据应用场景。

考虑到以上特点，加上本项目组成员已掌握的相关框架现状，项目组决定选择streamlit作为本项目的WebUI框架。

## 6.2 streamlit技术简介

streamlit 是一个用于快速创建数据应用程序的开源 Python 库。使用 Streamlit，可以使用简单的 Python 脚本构建交互式的数据应用程序，而无需编写大量的前端代码。它提供了一个简单易用的方式，使数据科学家和分析师能够快速创建漂亮的数据应用，展示数据分析结果和机器学习模型等。通过 Streamlit，可以使用 Python 编写代码来加载数据、可视化数据、添加交互元素（如滑块、下拉框等），并实时预览应用程序的输出。

Streamlit 的工作流程如下：

- 每次用户交互均需要从头运行全部脚本。
- Streamlit 根据 widget 状态为每个变量分配最新值。
- 缓存保证 Streamlit 重用数据和计算。

## 6.3 streamlit技术难点及解决说明

### 6.3.1 streamlit的组件(component)

本项目的streamlit界面如下图，主要分为侧边栏（sidebar）区域和正文区域。

在streamlit里，侧边栏区域固定在页面左侧。下图里sidebar里的组件有slider，button， checkbox等等。在代码里，使用with st.sidebar: 开头，就可以把组件放到侧边栏。这些组件在侧边栏里的布局顺序和代码里的顺序一致。

如果没有使用with st.sidebar: 开头，那么st的组件默认放到正文区域内。如下图，title，history和text_input都放在正文区域。除了text_input组件固定在正文区域最底部，其他组件在正文区域内的布局顺序和代码里的顺序也保持一致。

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDdiOGMxMjc0YTVlYzI4OTc5MjIyZGM3ZjM0ODQwNDBfeHUycXhsV1ZtU0xxb3dyd3kzQkp4dWVETVh3N1RsSmFfVG9rZW46VkdkVGJJSjF0bzIxOXV4dVBPNmNJajBYblpkXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

图 12 streamlit页面

### 6.3.2 streamlit刷新机制和session_state

streamlit 刷新机制是基于事件驱动的，它使得应用程序能够在用户与应用程序交互时动态更新。当用户与应用程序中的交互式组件（如滑块、下拉框等）进行交互时，Streamlit 会检测到这些事件，并触发相应的代码重新执行和页面更新。

具体来说，刷新机制可以简述如下：

1. 用户交互触发事件：当用户与应用程序中的交互式组件进行交互时（例如拖动滑块、选择下拉框、点击按钮等），会触发相应的事件。
2. 代码重新执行：与触发事件相关联的代码段将被重新执行。这些代码段通常与数据处理、可视化或其他应用程序逻辑有关。
3. 页面更新：一旦代码重新执行完成，Streamlit 将生成新的页面，反映出最新的状态和数据。这包括更新可视化图表、改变交互式组件的状态等。
4. 实时预览：用户可以在浏览器中实时看到页面的更新，从而与最新的应用程序状态进行交互。

这种基于事件的刷新机制使得 Streamlit 应用程序能够以动态和交互式的方式响应用户操作，为用户提供流畅的体验。

为了记录用户和streamlit后端交互的中间过程，streamlit有一个session_state的类，可以储存每次页面刷新后程序里的中间变量。session_state的使用方法和python里的dict类型类似。例如聊天历史一般存储在st.session_state.messages里，每次页面刷新后展示聊天历史都是从st.session_state.messages里把数据取出来展示到前端页面上。

### 6.3.3 streamlit常见问题及解决方案

项目组在使用streamlit进行应用编码的过程中，遇到了一些常见的问题，分别产生了一些解决方案，现在整理出来，以供参考。

**问题描述：**

每次streamlit刷新时都会重复加载文生图模型，耗时较长

**问题分析：**

每当用户交互或者页面刷新时，streamlit都会从头执行app入口函数，于是重复加载了一遍文生图模型。

**解决方案：**

在加载文生图模型的函数前面加一条@st.cache_resource 装饰器，streamlit会把该函数的返回结果保存到缓存里，当这个函数的参数不变时，就会直接返回缓存里的返回结果。

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=NTE4ODRhNTA5ZjMxNjFlYjNmZDA5MDYwZTI1OTgxZDBfMnlJVnF4ejJjYmtjQzFDcUNKa1liZUhGeXFoZklUY1JfVG9rZW46UEwyRmJFSFVFbzZaOXJ4MERnQmNvZ0I0bkdlXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

**问题描述：**

历史生成的图片在网页上没有显示

**问题分析：**

因为streamlit刷新时都会从头执行入口函数并且按代码里的顺序展示内容，要正常显示聊天历史里生成的图片，需要在对话历史里记录每个语句生成图片的文件位置，并且展示对话历史时如果处理该语句使用了文生图功能，需要在展示模型文字回复后展示历史生成的图片。

**解决方案：**

如果开启了文生图功能，在st.session_state.messages里加入生成图片的文件位置。

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=NjRjZWU3MjlkZjc3NjljYTkzN2JlNmRiYThmNTU2MDhfYnRFZTRuQm9qNlQxbHdRUFpBY1N5WnJ1MXhuUlJ3RFFfVG9rZW46UWwzVWJSZmxRb0xJUEt4TWttd2NTUnBmbnRkXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

然后，在展示对话历史时，如果历史信息里有图片路径，在输出文字回复后，使用st.image展示图片。

![img](https://hjjmcgs24u.feishu.cn/space/api/box/stream/download/asynccode/?code=OWIwYTJjMjA0Mjc3MjJkNmRhMDNkM2U3ODg3OTc0NmVfQTdST01oNlRoNFRsR3pxemIyM3luMmhxNjI5RFM4b3VfVG9rZW46SXNyZ2J1SGs3b2NsVHR4QzNOVWN5WTBFbjZkXzE3MTA0NTk3Mjg6MTcxMDQ2MzMyOF9WNA)

**问题描述：**

语音输入时，当用户点击“Clear Chat History”按钮，应用会将上次语音输入的识别结果提交给模型处理。

**问题分析：**

在点击“Clear Chat History”按钮后，streamlit会先执行该按钮的触发事件，然后从头执行启动脚本的入口函数。因为上次的录音缓存还保存在页面上，语音识别函数会自动识别录音并提交识别文字结果到模型。

**解决方案：**

要修复这一问题，需要记录当前录音缓存是否已经被识别过，但是项目里用的录音模块audiorecorder没有这个功能。后来参考https://github.com/B4PT0R/streamlit-mic-recorder这个库的做法，将上次录音音频数据的base64编码保存到session_state里，每次streamlit刷新后读取音频时先检查该音频的base64编码是否和上次录音音频数据一致，如果一致则不进行处理，这样就不会重复识别音频了。

## 6.4 参考

1. **streamlit** **[官网](https://docs.streamlit.io/get-started)** **[github](https://github.com/streamlit/streamlit)**
