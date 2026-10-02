# D33 / A8：冻结D29候选的浅层截断重放（predict-only，监督方已批准实现与执行）

状态：监督方科学决定，属Stage1过拟合研究的机制诊断；D32已完成。只读冻结产物、只做预测，不新拟合；无深度网格、无按E3选择、无新E4/地图、无Stage2/3/最终评价/close。审计run ed632775/base/session不变。

## 1. 问题与边界

D29从root到完整树在E3上纠正116、破坏134个root结果（`research/d29-error-changes.md`），按最终深度分组非单调。问题：同一行上，若在第一次被接受的root分裂后停止，与root及完整树相比如何？这是对已暴露开发案例的**事后机制诊断**，不是泛化成功，也不是干净的因果分解：区域大小、边界与自适应搜索共同变化；净持平可能掩盖相反的行级效应。

## 2. 固定比较对象（重放前预先固定）

对六个冻结D29候选（producer ab1ac83，运行`geoxgb-d29-confirm-20261001`）：
1. **root**：现有root booster（已存在的无增量对照，不新拟合）。
2. **depth1**：root分裂接受后的状态：区成员取`s_branch.pkl`的`"0"`/`"1"`列；booster取最终`xgb_0.ubj`/`xgb_1.ubj`（其SHA等于saved_log末次记录；含root决策时保留父/root副本的一侧，如H12 2020-10的`1`）。
3. **full**：冻结完整树（原路由与终端checkpoint）。

路由：depth1用`get_X_branch_id_by_group`配仅含`""`/`"0"`/`"1"`列的截断`s_branch`；不在两列中的区（无搜索、仅目标、仅C）走root（D32语义：空间`s-1`，预测root）。

## 3. 重建与强制门槛

- 用最小独立脚本，按现有准备/`main`语义重建S/C/E3特征（同快照、schema、r80 split seed 42、confirmation seed 42、keep掩码、键）；数据构造使用固定producer ab1ac83的代码（如经`git archive`导出），冻结Windows Python 3.12.10/固定包；不做通用重构。**澄清（监督方认可的比例方案）**：重放从已提交的当前包导入重建helper，前提是每个重建依赖与ab1ac83源码等价（逐文件/AST证明见`research/d33-source-equivalence.md`），并且真实数据完整树门槛精确通过；不导入归档helper，也不建运行时完整性框架。
- **先做完整树重放门槛**：root与full在C与E3上的概率（凡已保存）与原值一致；全部已保存S/C/E3硬预测与路由（branch_id）一致。S文件只有硬标签，不承诺S概率相等。任一不一致即停止解释、报告差异。

**证据保存（批准附加）**：逐键保存S/C/E3行（area、日期、H、真值、root/depth1/full硬预测与概率、路由）；新S概率为重放所得，不是历史保存证据。用现有helper记录冻结输入/checkpoint/源producer身份、重放脚本提交与运行时。精确相等检查用round-trip浮点解析，不静默放宽容差；完整树门槛必须先于浅层解释。核验`"0"`/`"1"`末次保存属于root决策并保留其保留父选择。原D29产物只读；不产生新E4或可供Stage2使用的地图。

## 4. 报告（逐H/T）

S/C/E3各自对root、depth1、full：四类混淆、危机TP/FP/FN、危机F1（主）、四类macro F1（次）；配对变化depth1−root、full−depth1（F1差及行级纠正/破坏、新增/丢失TP、新增/移除FP）；depth1两侧大小、保留父侧标记、booster SHA、按D32状态的支持计数。S为自适应复用，仅描述；C诊断；E3为已暴露开发目标的描述。六个相关候选，不推广。

## 5. 检查与流程

最小测试：截断路由与depth1成员一致、两列外的区走root、root副本侧使用root booster；门槛作为脚本内硬检查。规划提交→原绑定Claude native trellis-implement→native trellis-check→提交→predict-only重放（无拟合）→独立核验→文档；停止，交监督方综合。
