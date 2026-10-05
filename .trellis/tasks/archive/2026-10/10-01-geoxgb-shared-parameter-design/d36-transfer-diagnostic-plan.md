# D36：已有Stage1预测的转移失败分解（批准，仅分析，无新拟合）

D34/D35已完成并独立核验；本轮只解释C历史收益与E3微弱收益的差异，不推进Stage2/3或close。输入冻结为D34 `geoxgb-d34-e1-brier-20261002`（7b2bf6f）与D35 `geoxgb-d35-global-increment-20261002`（be5f485）。21根及所有C/E3行保留，不根据结果删日期。

## 分析口径（查看本轮分解结果前固定）

- C/E3按各行(area,target_month,H)与D34 snapshot连接，仅读取<=2020-12行。真值必须相同、键一对一；D35 root/global20/hard-local/Brier-local硬预测沿用已核验产物。
- Persistence严格取该行O=T−H的真实标签，`hist_phase_o00−1`得到四类code；不是`hist_latest_observed_phase`。起点缺失单独保留；做persistence分数比较时所有模型限制到同一非缺失键，并报告覆盖率。已有dev_baselines覆盖的15折用于交叉核对，不打开最终期ledger。
- 按起点危机状态与目标危机真值分为00持续非危机、01危机发生、10危机缓解、11持续危机、起点缺失。**这是事后误差分组，不是可部署路由或选择规则**；不可把目标真值用于模型启用。
- 首要分析为Brier-local相对root：逐组N、TP/FP/FN/TN、增加/减少TP与FP、修正/破坏的危机硬判断，以及原始四类argmax危机F1；全组计数须精确加回总量。单一真值层的F1不能作为跨组公平比较，优先看计数与组内错误率。
- 同时在相同C/E3行计算root、global20、hard-local、Brier-local的crisis Brier损失（概率取p3+p4or5），区分概率质量和argmax危机决策。概率只评分、不更改硬预测定义、不寻阈值。
- 汇总全体、逐H、逐目标，并作country层描述以发现收益/损失集中；不据此筛国家、日期、区域或参数，不作独立样本显著性宣称。跨fold重复键是不同模型预测，不声称独立观测。

## 产出与边界

监督方主导分析；脚本及CSV/JSON可先写到现有外部`geoxgb_runs`研究目录，最终小型研究脚本与解释记录落盘task/research并由Claude只读交叉审查。核验键/计数回加、已有15折persistence一致、原D34/D35全体分数复现。不改生产符号、已完成run或已有模型；无新拟合、概率阈值调优、候选筛选。PRD/implement只更新短指针和研究状态。

判读须保留替代解释：C与E3时间/事件分布不同，不能把差距唯一归为过拟合；分区总容量与空间结构仍混杂。此轮结果决定下一项有限实验，不自动启动新实验。

**结果（已完成，仅分析）**：见[research/d36-transfer-findings.md](research/d36-transfer-findings.md)；脚本`research/d36_transfer_diagnostic.py`，执行方独立核验与证据`research/d36_executor_*`。
