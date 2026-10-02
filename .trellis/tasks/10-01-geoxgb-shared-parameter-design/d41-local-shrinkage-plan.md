# D41 / A15：冻结分区的局部增量减半对照

2026-10-02，监督方依据用户委托决定此有界Stage1研究。Claude已只读核查产物可用性并提出限制；本契约评审、提交后才数值运行。零新增拟合，不修改生产预测规则，不运行Stage2/3、完整648、最终期或close。

## 问题与可识别范围

D34的Brier搜索候选在C上改善明显，E3收益很小且危机概率Brier略差。检验固定地图和模型后，将局部修正向同一root收缩是否改善跨期预测。仅改变输出增量幅度；不把它解释为学习率减半训练，不重新搜索地图、不重新做E2接纳。

地图/接纳在全量增量上选择，C主要是历史窗口内插值，G/本方案又经历开发选择。因此这是固定地图的敏感性对照，不能证明搜索过拟合已解决；E3结果接近也不能证明局部增量没有过拟合。21个相关开发折不作独立样本显著性推断，不自动采用任何臂。

## 冻结输入和最小预检

- 仅D34的21个Brier候选，producer `7b2bf6fe482d0a77a664f3627e493934d976696d`，运行`C:\Users\swl00\geoxgb_runs\geoxgb-d34-e1-brier-20261002`；H=4/8/12，T=2018-06/2018-10/2019-02/2019-06/2019-10/2020-02/2020-06。保留全部C/E3键，不纳入hard候选、D38 root、D40阈值或新模型。
- C使用候选`confirmation_predictions.csv.gz`中的`p_root_*`/`p_final_*`；E3按area连接候选`target_predictions.csv`的`p_partitioned_*`与匹配root的`root_target_predictions.csv`中`p_pooled_*`。验证唯一键、完整双向键集合、真值和保存argmax一致；不能依赖行序。记录输入hash。
- 根据现有checkpoint/continuation记录核查实际使用的局部模型均以本root为共享来源、只增加L1 20轮；root/回退路由保持root。具名分支也可能只是root副本：`kind=fresh`、`actual_local_rounds=0`且记录/实际UBJ hash均等于root时，归为零增量，不因`terminal_branch`就当作局部模型。复用D34/D35已核验证据与元数据，不新建追溯框架。发现不符合则停止，不偷偷跳过候选。
- 读取概率使用float64和round-trip解析；要求全部有限、严格正、四类和在float32输出容差1e-6内。遇到零/不合法值停止并报告，不能加入epsilon。保存原概率，禁止为原root/full重归一化而改变既有分数。
- exact-origin persistence及00/01/10/11/缺失起点分组复用D36的连接口径与已核验行来源；如需读取snapshot，仅必要列且过滤target<=2020-12。不读取最终期ledger。比较persistence时各臂用同一非缺失键；分组使用目标真值仅作事后诊断。

## 唯一干预与数值定义

理想margin为`m_half = m_root + 0.5*(m_local-m_root)`。softmax下等价于归一化几何均值。本轮直接在保存概率上定义：

    a = sqrt(p_root * p_local)  # float64，固定四类轴1/2/3/4或5
    p_half = a / a.sum(axis=1, keepdims=True)

保存float32概率的舍入意味着该变换不是原booster raw-margin的逐位重放；明确称保存概率上的几何收缩。若某行root与local四概率完全相同，half直接复制root，保留零增量/回退精确不变。唯一新臂alpha=.5；alpha=0/1只对应已有root/full，不作网格、不挑日期/国家/分区、不寻阈值。

各臂均四类argmax后折叠crisis（code>=2），并列遵循固定轴首类；危机概率为p3+p4or5。先固定全体half概率/标签再评分，truth不进入变换。危机Brier作为首要连续诊断，原危机F1仍报告，不替换任务主终点。

## 输出、检查和停止

- C/E3分别输出全体、同键persistence子集、逐H、逐root（H/T）、四个转移组及缺失组的N、Brier、TP/FP/FN/TN与危机F1；明确区分汇总混淆F1和逐折均值差。另按已冻结模型来源报告真实局部路由/零增量路由的N及同行三臂Brier/混淆，避免零增量行稀释整体delta；这是固定路由分层，不是事后筛选。相对root的修正/破坏、增加/减少TP/FP按组回加；记录half/full之间的硬判断翻转数量及概率差幅度。不给单真值分组的F1作跨组优劣解释。
- root/full的原始C/E3混淆和Brier复现D36；报告逐折delta，不能只呈现最佳汇总。无E4或下游地图输出。保留包含root/part/area/target/H/truth/route及四类概率/预测的逐键外部文件、紧凑summary和输入hash。
- 一个最小外部脚本，冻结Windows Python3.12.10/numpy2.2.6/pandas2.2.3；不改生产符号。自检用合成margin验证几何均值恒等式、相同分布直接复制、无truth依赖；实际数据核对完整键、原臂分数、fallback不变及分组回加。执行方与监督方用不同计算表达式独立核对（log-probability平均再softmax），按浮点容差核对概率、精确核对argmax和混淆；若近并列导致标签差异须报告。
- 小脚本/摘要/核验/综合最终留task research，大逐键文件留外部并记录路径/hash。先提交spec并同步prd/design/implement/PROGRESS与experiment-plan短指针（保持文件大小限制），再执行。结果后停止扩展，由监督方判断下一方向；不继续扫描alpha，不将本轮完成写成Stage1过拟合解决。
