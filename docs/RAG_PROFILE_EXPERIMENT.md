# RAG Profile 优化、评测与回退

## 当前结论

线上活动 Profile 保持 `current`。首轮同数据、同 PDF、同 Parser、同模型的离线 A/B 结果为 `No-Go`：optimized 的点估计更高，但 20 个问题上的 MRR/NDCG 提升尚未达到统计明确，检索 p95 又从 3339 ms 增至 6151 ms（1.842 倍），超过 1.25 倍门槛。因此没有运行后续付费回答评测，也没有切换线上链路。

权威产物：

- 当前基线：[rag-current-a-v1.json](../eval/baselines/rag-current-a-v1.json)
- 首轮 A/B 决策：[rag-profile-ab-manifest.json](../eval/reports/rag-profile-ab-screen20-final-v2/rag-profile-ab-manifest.json)
- Control 明细：[rag-profile-a.json](../eval/reports/rag-profile-ab-screen20-final-v2/rag-profile-a.json)
- Experiment 明细：[rag-profile-e.json](../eval/reports/rag-profile-ab-screen20-final-v2/rag-profile-e.json)

## 链路和隔离边界

```text
current（默认、原始链路）
  原切块 + 原索引身份 + 原检索行为

optimized（仅实验或已通过门禁的 canary）
  A  原始链路
  B  + 标题/章节增强 retrieval_text
  C  + Parent-aware 二次重排
  D  + 同 Parent 多样性
  E  + 结构感知混合切块
```

`current` 继续使用原身份：

```text
document_id       = research-paper:{scope_id}:{paper_id}
knowledge_base_id = research-scope:{scope_id}:v{scope_version}
```

optimized 使用独立身份，例如：

```text
document_id       = research-paper:{scope_id}:{paper_id}:rag:optimized-e
knowledge_base_id = research-scope:{scope_id}:v{scope_version}:rag:optimized-e
```

因此实验建库不会替换 current 文档版本。检索、Parent 查找、Evidence 列表、Evidence 读取和引用验证都按 Profile/消融阶段过滤。引用仍指向原始 Child 文本，增强后的 `retrieval_text` 只参与召回与重排。

## 安全配置

默认配置：

```ini
TASKFORGE_RAG_ACTIVE_PROFILE=current
TASKFORGE_RAG_EXPERIMENT_PROFILE=current
TASKFORGE_RAG_OPTIMIZED_ABLATION=e
TASKFORGE_RAG_EVALUATION_MODE=false
TASKFORGE_RAG_OPTIMIZED_PROMOTION_MANIFEST=
```

离线实验必须同时设置活动与实验 Profile，并开启 evaluation mode。线上启用 optimized 时 evaluation mode 必须关闭，且必须提供通过全部检索、回答和引用门槛的 promotion manifest。缺少 manifest、manifest 为 No-Go，或回答/引用评测未完成时，配置校验会拒绝启动 optimized。

## 冻结 current 基线

```powershell
.\.venv\Scripts\python.exe scripts\freeze_rag_current_baseline.py
```

基线清单记录 Git commit、dirty paths、关键代码哈希、模型和包版本、切块参数、Candidate@50、Top@8、MinerU 设置、数据集/PDF 清单哈希，以及 current/A 的复现结果。

## 运行 A–E 消融

先只检查运行计划：

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_profile_ab.py --dataset .taskforge\eval-cache\qasper-dev-v0.3.json --split eval\splits\qasper-dev-clean-holdout-100-v2.json --pdf-manifest .taskforge\eval-cache\qasper-clean-holdout-real-pdfs-v3.json --output-dir eval\reports\rag-profile-ab-next --state-root .taskforge\eval-runs\rag-profile-ab-next --limit 100 --pdf-parser-backend mineru --mineru-base-url <URL> --mineru-expected-version 3.4.4 --plan-only
```

去掉 `--plan-only` 执行真实评测。每个阶段必须使用空的独立 state 目录；脚本遇到非空目录会拒绝运行，防止 A/E 互相污染。固定条件包括同一输入哈希、Candidate@50、Top@8、原 Query、禁用图检索和知识图谱、相同 Embedding/Reranker/Parser。

门禁同时要求：

- Gold 对齐通过；
- Recall@5、Recall@10、Agent-visible Recall@8 不下降；
- MRR 或 NDCG@8 的 paired bootstrap 95% CI 下界大于 0；
- 引用定位、精度和真实 roundtrip 验证不下降；
- 表格、列表和章节子集不下降；
- optimized 索引身份独立；
- p95 不超过 current 的 1.25 倍；
- 在检索门禁通过后，配对回答正确率、严格 grounded accuracy、引用有效率和 Gold-page 引用精度不下降。

检索门禁已失败时可以提前停止付费回答评测；此时结论只能是 No-Go。

## 首轮对照结果

| 指标 | current A | optimized E | 门禁 |
| --- | ---: | ---: | --- |
| Recall@5 | 0.6567 | 0.7233 | 通过 |
| Recall@10 | 0.7900 | 0.8733 | 通过 |
| Agent-visible Recall@8 | 0.7567 | 0.7733 | 通过 |
| MRR | 0.4945 | 0.5484 | CI 未明确 |
| NDCG@8 | 0.5029 | 0.5524 | CI 未明确 |
| 引用定位@8 | 0.8500 | 0.8500 | 通过 |
| 引用精度@8 | 0.1250 | 0.1313 | 通过 |
| 引用 roundtrip | 1.0000 | 1.0000 | 通过 |
| 检索 p95 | 3339 ms | 6151 ms | 失败（1.842×） |

MRR 差值为 +0.0539，95% CI 为 `[-0.0582, 0.1701]`；NDCG@8 差值为 +0.0494，95% CI 为 `[-0.0366, 0.1378]`。区间跨 0，不能据此断言 optimized 确定优于 current。

## 回退与验证

回退只需恢复活动 Profile，不需要删除 optimized 索引：

```ini
TASKFORGE_RAG_ACTIVE_PROFILE=current
TASKFORGE_RAG_EXPERIMENT_PROFILE=current
TASKFORGE_RAG_EVALUATION_MODE=false
TASKFORGE_RAG_OPTIMIZED_PROMOTION_MANIFEST=
```

重启检索服务以清空进程内检索缓存，然后先运行静态路由检查：

```powershell
.\.venv\Scripts\python.exe scripts\verify_rag_current_route.py
```

再用一个已入库 Scope 做真实索引冒烟：

```powershell
.\.venv\Scripts\python.exe scripts\verify_rag_current_route.py --scope-id <SCOPE_ID> --tenant-id <TENANT_ID> --user-id <USER_ID> --query "<FIXED_SMOKE_QUERY>" --expected-evidence-id <KNOWN_CURRENT_EVIDENCE_ID>
```

脚本只有在配置解析为 `current-a`、原索引身份不变、查询结果全部来自 current corpus 且预期证据存在时才返回退出码 0。optimized 索引可以保留用于后续实验，但不会被 current 查询、展示、读取或引用。

## 单独测试 retrieval_text

在完整 E 链路 No-Go 后，又使用锁定的 100 个问题单独比较了：

```text
current A
vs
optimized B = A + 标题/章节 retrieval_text
```

本轮没有启用 Parent-aware、多样性、结构感知切块、查询扩展、图检索或知识图谱。输入数据、PDF manifest、MinerU 版本、Embedding、Reranker、Candidate@50 和 Top@8 保持一致；B 使用独立 `optimized-b` 索引。

| 指标 | current A | optimized B | 差值 |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.7022 | 0.7288 | +0.0267 |
| Recall@10 | 0.8447 | 0.8647 | +0.0200 |
| Agent-visible Recall@8 | 0.7938 | 0.8163 | +0.0225 |
| MRR | 0.5611 | 0.5881 | +0.0270 |
| NDCG@8 | 0.5767 | 0.5997 | +0.0229 |
| 引用定位@8 | 0.8600 | 0.8500 | -0.0100 |
| 引用精度@8 | 0.1300 | 0.1300 | 0 |
| p95 | 8073 ms | 8467 ms | 1.049× |

MRR 的 paired bootstrap 95% CI 为 `[-0.0053, 0.0614]`，NDCG@8 为 `[-0.0073, 0.0545]`，均跨 0。章节类问题的 Visible Recall@8 从 0.8167 降到 0.7889。因此 B 虽然平均召回更高且延迟成本可接受，仍未满足“排名明确提升、引用不下降、子集不下降”的门槛，结论继续为 `no_go_keep_current`。

详细报告：[retrieval_text A/B comparison](../eval/reports/rag-profile-retrieval-text-locked100-v1/comparison.json)。

## Contextual Child 初筛

随后测试了单次 Cross-Encoder 重排窗口：候选仍使用原始 Child 文本，重排输入临时组合同一 Parent 下的前邻居尾部、当前 Child 和后邻居头部；不拼文档标题、不运行 Parent-aware，引用仍定位当前 Child。

锁定 20 问题初筛结果：

| 指标 | current A | Flat 2000 | Contextual Child |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.6567 | 0.7567 | 0.7067 |
| Recall@10 | 0.7900 | 0.8317 | 0.7733 |
| Agent-visible Recall@8 | 0.7567 | 0.8192 | 0.7233 |
| MRR | 0.4945 | 0.5350 | 0.5218 |
| NDCG@8 | 0.5029 | 0.5949 | 0.5292 |
| 引用定位@8 | 0.8500 | 0.9000 | 0.8000 |
| p95 | 3339 ms | — | 3578 ms |

邻居上下文带来 Recall@1/5 的局部收益，但 Recall@10、Visible Recall 和引用定位均低于 current，更明显低于 Flat。按早停规则未运行 100 问题，实验能力保持默认关闭；报告见 [Contextual Child comparison](../eval/reports/rag-profile-contextual-child-screen20-v1/comparison.json)。

## 下一轮建议

当前不切换任何优化 Profile。若继续研究 retrieval_text，应先定位章节子集和引用定位退化的具体问题，并调整标题/章节拼接格式；只有新的检索统计门禁通过后，才运行配对回答/引用评测并考虑小范围 canary。图检索和知识图谱不在本轮范围内，当前失败点也不需要靠它们解决。

## Boundary-aware Flat 初次评测

在保持 Flat 2000/0 粒度、BM25、百炼 `text-embedding-v4`、RRF、`qwen3-rerank` 和 paper scope 不变的条件下，新增了隔离策略 `boundary_aware_flat_v1`。该策略只把边界移动到结构安全位置：目标 2000 字符、最小 1000、最大 2600、边界搜索窗口 400；标题使用保守的编号/常见顶层识别，表格、公式、图和列表不从内部切断。

该策略在相同 60 篇论文、177 个问题上完成了 MinerU 全量评测。结果为 Recall@10 `0.9104`、Recall@50 `0.9763`、MRR@10 `0.6227`、NDCG@10 `0.6393`；冻结 Flat 基线分别为 `0.9262`、`0.9763`、`0.6366`、`0.6551`。英文 Recall@10 上升到 `0.9153`，但中文 Recall@10 下降到 `0.9056`。同批次 Control 与 Candidate 的 P95 比例为 `1.084x`，但质量门禁已经失败，因此该策略结论为 `No-Go`，不进入线上。

正式对比报告：[boundary-aware Flat comparison](../eval/reports/mixed-mineru-boundary-flat-30zh-30en-bailian-paper-comparison-v1.json)。Candidate 的独立全量报告：[boundary-aware Flat full report](../eval/reports/mixed-mineru-boundary-flat-30zh-30en-bailian-paper-final-v1.json)。原 Flat 链路和索引继续作为默认及回退路径。
