# TaskForge 评测说明

本文说明 TaskForge 当前如何评测论文发现、全文证据检索和研究报告生成。技术指标名与命令保留英文，说明文字使用中文。

## 评测目标

TaskForge 把评测分为三个独立任务：

1. 开放论文发现：能否从外部来源找到相关论文；
2. 已选论文全文检索：能否从用户确认的论文中找到正确证据；
3. 报告生成：能否基于证据生成可核对的结论和引用。

三类任务的搜索空间、输入和指标不同，不能把一个任务的结果当成另一个任务的成绩。

## 数据与版本规则

正式结果必须记录：

- 数据集、split 和问题 ID；
- PDF、标注和查询文件的 SHA-256；
- 解析器及版本；
- 分块参数；
- embedding 和 reranker；
- Candidate、Top-K 和查询改写预算；
- 代码提交；
- Provider、模型和运行时间；
- 错误、重试和回退情况。

调参集、验证集和最终冻结集必须分开。修改语料、标注、解析、分块或 Scope 后，不能继续沿用旧结果。

## 开放论文发现

开放发现评测输入是用户研究问题，输出是候选论文列表。主要关注：

- `Recall@K`：已知相关论文有多少进入前 K；
- `Precision@K`：前 K 中有多少属于已知相关论文；
- `nDCG@K`：相关论文是否排在更靠前的位置；
- Provider 成功率和限流情况；
- 中文、英文和混合查询的结果差异。

开放发现只评估论文候选，不评估 PDF 内部证据。中文通道、英文改写和语言偏好需要在同一冻结数据上成对比较。

## 已选论文全文检索

这项评测从用户已经选定论文开始。每个查询只能检索对应论文或 Scope 内的全文，不能搜索整个语料库。

主要指标：

- `Recall@K`：标注证据是否进入前 K；
- `MRR@K`：第一条正确证据出现得是否足够早；
- `nDCG@K`：多条证据的整体排序质量；
- `Candidate@50`：正确证据是否进入重排候选池；
- Scope 违规数：是否返回了范围外论文；
- p50 / p95：检索耗时。

### 当前冻结基线

正式 selected-paper 基线为 [`paper-scoped-flat-bailian-v1.json`](../eval/baselines/paper-scoped-flat-bailian-v1.json)。

数据范围：

- 30 篇中文真实 PDF；
- 30 篇英文真实 PDF；
- 177 个标注问题。

固定链路：

```text
MinerU 3.4.4
→ Flat 2000 字符 / 0 overlap
→ BM25
→ 百炼 text-embedding-v4
→ RRF
→ 百炼 qwen3-rerank
→ 单论文 knowledge_base_id 过滤
```

结果：

| 指标 | 结果 |
|---|---:|
| 整体 Recall@10 | 92.62% |
| 中文 Recall@10 | 95.00% |
| 英文 Recall@10 | 90.15% |
| Recall@50 | 97.63% |

这个基线只适用于“用户已经选定论文后的全文问答”。全库检索拥有更大的搜索空间，不能与它直接比较。

### 运行配置与冻结基线

当前本地运行使用百炼 embedding 和 reranker，但分块配置为 `parent_child`；冻结基线使用 `Flat 2000/0`。因此：

- 可以把 92.62% 作为当前正式评测基线；
- 不能直接把该数字解释为当前 `parent_child` 在线索引的实测成绩；
- 切换分块方式后必须重建索引，并用同一数据重新评测。

## 报告生成评测

报告生成不只检查文字相似度，还检查：

- Planner、Evaluator、Writer、Critic 是否全部完成；
- 最终状态是否进入 `waiting_human_review`；
- 主要结论是否绑定 Evidence ID；
- Evidence ID 是否属于当前 Scope 和版本；
- 引用是否能回到正确论文、页码和原文；
- 是否出现无证据结论；
- Token、耗时、重试和 Provider 错误。

当前真实模型四角色链路已经完整跑通。历史配对评测中，端到端答案 Token F1 相对初始基线提升 36.35 个百分点；README 和本文不使用绝对答案分数作为当前产品宣传指标。上下文与交接优化后，同任务 Token 消耗下降约 71%。

## 证据质量评测

除了排名指标，还需要检查返回内容是否适合引用：

- 参考文献目录不得作为主要证据；
- 标题、作者邮箱、关键词包装不得冒充正文；
- 模板和占位表格必须被过滤；
- 同一内容不得以多个重复证据卡返回；
- 短版展示和完整原文必须保持同一个 Evidence ID；
- Writer 和 Critic 读取的原文必须与用户看到的来源一致。

## 对比实验规则

候选方案要替换当前基线，必须满足：

1. 使用相同论文、问题、标注和解析缓存；
2. 只改变本次要验证的变量；
3. 保持 Scope、ACL 和全文状态过滤；
4. 记录每项中英文指标；
5. 不能用平均值掩盖中文或英文明显回退；
6. 不能把 smoke test 当成正式提升；
7. 产物必须包含配置、哈希和代码版本。

## 复现 selected-paper 基线

以下命令会调用百炼 embedding 和 reranker，可能产生费用：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_mixed_dual_mineru.py `
  --mode flat `
  --scope paper `
  --output eval\reports\mixed-mineru-flat2000-30zh-30en-bailian-paper-scoped-final-v1.json `
  --state-dir .taskforge\eval-runs\mixed-mineru-flat2000-30zh-30en-bailian-all-v1
```

运行前需要准备：

- `.env` 中的 `TASKFORGE_BAILIAN_API_KEY`；
- 锁定的中英文 split；
- 对应 PDF 和 MinerU 缓存；
- 足够的 Provider 配额。

## 工程回归

后端：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

前端：

```powershell
cd frontend
pnpm install --frozen-lockfile
pnpm build
```

Docker：

```powershell
docker compose config --quiet
docker compose ps
```

工程测试验证代码契约，不等于模型质量评测。模型成绩必须绑定数据、Provider、模型、Prompt、代码版本和预算。

## 评测产物位置

```text
eval/baselines/   正式冻结基线
eval/splits/      数据划分
eval/queries/     查询与标注
eval/reports/     评测报告
.taskforge/       本地缓存和未提交运行状态
```

PDF 解析和检索实现见 [PDF RAG 流程](PDF_RAG_PIPELINE.md)，完整产品链路见 [论文研究流程](PAPER_RESEARCH_AGENT.md)。
