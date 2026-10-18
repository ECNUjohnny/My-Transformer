# CS336 作业 PDF 中文翻译规范（译者必读）

你负责把 Stanford CS336 作业说明 PDF 的指定页范围翻译成中文 LaTeX 片段。
产出文件是整本书的一个部分，最终会与其他部分合并编译。

## 一、产出文件

- 写入指定路径（如 `translations/assignment1/part1.tex`），UTF-8 编码。
- 文件内容为 LaTeX **正文片段**：只能包含
  section / subsection / subsubsection 级标题、段落、公式、表格、代码块、插图、
  itemize / enumerate / description 列表、脚注。
- **禁止**出现在片段里：`\documentclass`、`\usepackage`、`\begin{document}`、
  `\maketitle`、`\tableofcontents`、`\title`、`\author`（这些在主文件里已经写好）。

## 二、读取原文

用 Read 工具读原 PDF，`pages` 参数填你负责的页范围（如 `"17-32"`）。
按页顺序翻译，不要跳过任何内容：正文、任务描述、交付要求（Deliverables）、
提示（Hints）、表格、脚注、要点列表全部翻译，禁止概括或省略。

## 三、LaTeX 片段示例（照此风格）

```latex
\section{语言建模}
\subsection{目标}
本作业的目标是……

交叉熵损失定义为
\begin{equation}
\mathcal{L}(\theta) = -\frac{1}{B}\sum_{i=1}^{B}\log p_\theta(x_i \mid x_{<i})
\end{equation}
其中 $B$ 是批量大小（batch size）。

\begin{itemize}
  \item 第一条要求……
  \item 第二条要求……
\end{itemize}

\begin{lstlisting}
def sgd(params, lr):
    params -= lr * params.grad  # 注释可译为中文
\end{lstlisting}

行内代码写成 \tm{model.forward(x)}，其中 \tm{forward} 表示前向传播。

\begin{table}[htbp]
\centering
\begin{tabular}{@{}llr@{}}
\toprule
数据集 & 词表大小 & 文档数 \\
\midrule
FineWeb & 50257 & 9.7M \\
\bottomrule
\end{tabular}
\caption{原文表格的中文说明}
\end{table}
\end{verbatim 示例结束——这只是说明文档，不要把本段复制进译文}
```

## 四、语法红线（编译会炸的地方）

1. 普通文本里的特殊字符必须转义：
   - `%` → `\%`（百分号最常见！）
   - `&` → `\&`（表格和公式对齐环境里的 `&` 除外）
   - `#` → `\#`
   - `_` → `\_`（文件名、变量名里的下划线！如 `trivial\_baseline`）
   - `<` `>` 在数学模式外直接用没问题，但在 `\tm{}` 里建议保持原样
2. lstlisting 环境内部是逐字环境，**不需要也不可以转义**。
3. `\begin`/`\end`、`$...$`、大括号必须严格配对。写完后自查一遍。
4. 不要使用模板未加载的宏包或环境。可用：amsmath/amssymb（equation、align、
   split、cases、matrix 系）、booktabs、tabularx、longtable、listings、
   graphicx、enumitem、hyperref（`\url{}`、`\href{}{}` 可用）。
5. 不要用 `\ref`/`\label` 做跨文件引用。原文写 "see Section 3.2" 就直接译
   "见第 3.2 节"；"see Equation (5)" 译 "见式 (5)"。
6. 中文与英文/数字之间留一个空格（LaTeX 会自动处理部分间距，手工留更保险）。

## 五、插图处理

`figures/manifest.json` 按页码列出了从原 PDF 提取的候选图片
（`kind=raster` 是位图，`kind=vector` 是矢量图渲染，矢量图清晰度更高）。

对你页范围内的每个候选图：
1. 用 Read 工具查看该图片文件（绝对路径见 manifest 同目录）。
2. 是有意义的插图（示意图、流程图、曲线图、架构图）→ 嵌入：

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{figures/p07_v03.png}
\caption{中文图注：按原图注翻译，或根据图的内容写一句说明}
\end{figure}
```

3. 是无意义内容（页眉 logo、表格线框残留、代码框残留、空白碎片）→ 跳过，不嵌入。
4. 图内的英文文字无法修改，保留原样即可；图注用中文。
5. 若原文该页明显有插图但 manifest 里没有对应文件，在相应位置写：
   `\textit{【图：见原 PDF 第 N 页】}`。

## 六、翻译风格

- 术语用机器学习社区标准中文译名：cross-entropy→交叉熵、learning rate→学习率、
  learning rate warmup→学习率预热、gradient clipping→梯度裁剪、
  embedding→嵌入、tokenizer→分词器、fine-tuning→微调、
  scaling law→缩放定律、throughput→吞吐量、latency→延迟、
  data parallelism→数据并行、tensor parallelism→张量并行、
  pipeline parallelism→流水线并行、sharding→分片、overhead→开销。
- 以下专有名词保留英文：Transformer、FlashAttention、ZeRO、AdamW、LAMB、
  Triton、CUDA、NCCL、FineWeb、C4、OpenWebText、GPT-2、Llama、TinyStories、
  Stanford CS336、pusheen（吉祥物名）、Hugging Face、Weights \& Biases（注意 `&` 转义）。
- token、logit、checkpoint、benchmark 等这类词：按中文 ML 社区惯例，
  token 保留英文，其余优先用中文（检查点、基准测试），首次出现可括注英文。
- 首次出现的关键术语可以写成「交叉熵（cross-entropy）」的形式。
- 代码标识符、文件路径、URL、超参数名、数据文件名一律不翻译。
- 数字、单位、公式保持原样。
- 语气：技术文档，准确平实；任务指令用「请实现……」「你需要……」。

## 七、自检与返回

写完文件后：
1. 通读检查语法配对（尤其是 `\%`、`\_`、`$`、大括号）。
2. 确认没有把禁止内容（`\documentclass` 等）写进片段。
3. 返回一段简短总结：负责的页范围、生成的 section 标题列表、
   嵌入的图片文件名及对应原文页码、跳过的图片数量。
