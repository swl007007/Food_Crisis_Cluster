# D39 / A13：现存概率的排序、校准与决策诊断（已完成，仅诊断；独立核验通过）

2026-10-02：监督方依据用户委托批准零新增模型拟合的分析；不改变终点、模型、地图或训练。不运行Stage2/3、最终期、完整648或close。原任务/审计run/执行会话保持。

## 问题与输入

D38同键E3：原root F1/Brier .551516/.101885；anchored .562155/.101291；输出调整控制 .583974/.115976；persistence F1 .586406。不能把F1差距简单视作概率更差，也不能从这些开发结果推导部署阈值。本轮分清已有概率信号、校准和固定四类argmax的影响，再决定是否需要改变训练。

- 唯一数值输入：已核验D38 `geoxgb-d38-persistence-margin-root-20261002` 的21个 `rows_E3.csv.gz` 及其summary/identity（producer `2d4fe4e`）；无模型拟合、无特征重建、无最终ledger或2021+标签/成绩读取。确认21组H/T、每折键唯一、truth/四类概率轴/argmax、同行persistence缺失；记录输入hash。
- 原root、anchored、posthoc、prior-only均保留，`p_crisis=p3+p4or5`。persistence仅在精确起点标签非缺失键比较，一律同键；所有root全键F1/Brier作为覆盖核对保留。缺失起点另报行数及基准结果，不回填。
- 分析限定暴露过的开发E3；C已有历史插值证据，本轮无需重复分析。主决策仍四类argmax折叠 code>=2，不部署任何新决策。

## 固定输出（不调参）

1. 每根、按H及整体：样本数、真实危机率、平均p、原决策危机率、危机F1/混淆、exact逐行Brier、ROC-AUC、average precision（AP，同时列真实危机率作无信息参考）。仅一个标签类别时AUC/AP记undefined，不伪造0或跳过整折。
2. 相同非缺失键上按起点危机状态（0/1）再报告上述概率排序指标，附正负样本数。persistence在任一组内为常量，AUC若两类皆有则.5、AP等于组危机率。解释为区分发生与稳定、持续与恢复的信号，不把组间排序当成组内信号。
3. 原/anchored/posthoc/prior-only固定10等宽概率区间 `[0,.1),...,[.9,1]` 的N、平均p、真实危机率、逐行Brier。按H及整体给表；空箱保留。无分位数自适应分箱、校准器拟合或Murphy分解。p的浮点边界仅用于箱号夹紧，Brier用保存概率原值。
4. 四类argmax危机判定 vs 固定 `p_crisis>=.5` 的2×2计数，并记录两种不同判定各自纠正/破坏数；可报告这一固定规则的F1作机制诊断，但不得据E3选择/采用该规则。不存在E3最优阈值扫描、最大F1、可部署增益或按国家/区域挑选。
5. 与D38已验证汇总F1/Brier核对；同键prior-only argmax==persistence、constant-persistence组内指标核对作为可运行检查。脚本直接复用numpy/pandas/sklearn已安装指标，不写新的ROC/AP实现用于主分析。

审阅补充：prior-only在每一起点组内也为常量，AUC=.5、AP=组危机率，独立核对这一恒等式。其起点缺失行恰为p_crisis=.5，固定>=.5与argmax的不同全来自tie规则（713行），须在结果中明确标注，不能当成学得的收益。persistence/prior-only的池化AUC/AP是两级score含ties的参考，数学上可比较，但不表明有组内排序或完整曲线支配。

## 证据与解释

主线程写一个外部分析脚本与JSON结果，由Claude独立抽查原始保存行、手工/独立实现AUC的成对排序含ties、AP和固定箱汇总；无需native模型实现或新测试框架。独立检查范围明确，不把少数抽查称全量核验。结果/脚本/解释留在task research，更新PRD/design/implement/PROGRESS短指针后提交；每次提交尝试GitNexus detect_changes，图不可用则据实际错误记录，不重建索引。

仅描述在这些E3上的排序与校准。高AUC/AP不保证一个起点前可确定的决策规则能胜过persistence；低F1也不唯一证明过拟合。跨折池化排序受日期混合影响，所以逐折与按H结果同时保留；这些折重叠且反复暴露，不作独立样本显著性推断。若随后研究决策规则，必须先另写设计，阈值/校准器仅从各预测起点之前的信息确定，再评估，不用E3调优。本轮不自动进入新训练。

## 结果（2026-10-02；无拟合）

见[research/d39-probability-findings.md](research/d39-probability-findings.md)；主脚本/结果`research/d39_probability_diagnostic.py/.json`，执行方独立核验`research/d39_executor_check.py/.json`。
- 独立核验范围：整体、各H及6个单根（每H首末2018-06/2020-06）×全键/同键/缺失/起点0/起点1×四模型（同键与分层另含persistence），2414项比较0不一致；其余15根只经整体与按H汇总核对。
- 同键排序：原AUC/AP .8695/.6522，persistence .7428/.4231（两级含ties参考，不表明曲线支配）。起点非危机组原.8033/.2408（危机率.0952），起点危机组.7456/.8442（.6059）。
- 固定mass>=.5使F1变差（.5378 vs argmax .5515）；argmax更宽松（1367 vs 71行不一致）。prior-only缺失起点713行恰为.5 tie，>=.5与argmax差异全来自tie规则。
- 固定箱：H8中高概率过度自信，H12整体及高箱不足；暴露E3的粗分箱不授权校准器。
- 监督方综合：排序信号存在；不证明终点规则单独造成差距，也不说明分区过拟合已解决。D38仍为探索性root信号、非默认。起点前的顺序阈值诊断仍为设计讨论，需另立spec。
