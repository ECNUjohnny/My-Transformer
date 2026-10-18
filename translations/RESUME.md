# CS336 翻译进度快照（2026-10-13）

接手前先读 `CONTRACT.md`（翻译规范）。原文 PDF 在各 assignment 仓库目录下。

## 状态总览

| 作业 | 原 PDF | 页数 | 状态 |
|---|---|---|---|
| assignment1 | cs336_assignment1_basics.pdf | 47 | part1(1–16)✓ part2(17–32)✓ **缺 part3(33–47)** |
| assignment2 | cs336_assignment2_systems.pdf | 48 | part1(1–16)✓ **缺 part2(17–32)、part3(33–48)** |
| assignment3 | cs336_assignment3_scaling.pdf | 8 | ✅ 完整（part1 覆盖全篇，末尾已到参考文献） |
| assignment4 | cs336_assignment4_data.pdf | 16 | ✅ 完整（part1 覆盖全篇） |
| assignment5 主册 | cs336_spring2026_assignment5_alignment.pdf | 39 | part2(14–26)✓ part3(27–39)✓ **缺 part1(1–13)** |
| assignment5 补充 | cs336_spring2025_supplement_safety_rlhf.pdf（实际为 spring2026 文件） | 18 | ✅ 完整（part1 覆盖全篇） |

## 待办（按优先级）

1. `assignment1/part3.tex` —— 原文第 33–47 页。原文纯文本已提取好：
   用 `translations/_p33_47_clean.txt`（UTF-8 清洗版），不必再读 PDF。
   写完后在 `assignment1/main.tex` 的 `\input{part3}` 下即可编译。
2. `assignment5/part1.tex` —— 原文第 1–13 页（part2 从第 14 页开始，注意衔接，
   part2 开头是"接上一页代码清单的结尾"，即原 PDF 第 13→14 页有一段跨页代码，
   翻 part1 时要把第 13 页的代码清单写完整）。
3. `assignment2/part2.tex` —— 原文第 17–32 页（从 PDF 重新读）。
4. `assignment2/part3.tex` —— 原文第 33–48 页（从 PDF 重新读）。

## 全部补齐后

- 逐个编译各 `*/main.tex`（xelatex，两遍），确认无错、生成完整 PDF。
- 每个作业已有 `check_partN.pdf` 语法抽查通过的历史，新写的 part 建议照旧
  做一次片段编译自检再并入 main。
- assignment2 源目录有一个重复下载的 `assignment2-systems-...（1)` 文件夹，
  内容相同，任选其一即可。

## 环境备注

- 平台 win32 / Git Bash；路径示例：`/d/WorkSpace/Project/My Transformer/translations/assignment1/part3.tex`。
- figures 从各作业 `figures/manifest.json` 查，图片用 Read 工具看内容再决定是否嵌入。
