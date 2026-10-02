# D37 / A11：固定时间权重的root对照（已完成；不采用；监督方独立核验通过）

监督方依据用户委托的一阶段探索权限批准本轮。沿用任务及审计run ed632775/base/原Claude会话。D36已完成：C收益不能代表未来转移；D34的局部E3增益远小于root对persistence的差距。先检验root的时间权重，不改分区。无Stage2/3、full648、最终评价或close。

## 1. 冻结假设与范围

假设：59个月内较老观测对当前预测起点的适用性较弱；保留全部行，但适度提高近期行权重，可能改善root外推。D36只支持研究这个假设，不证明分布变化是唯一原因。选择24个月半衰期是事前固定的有限探索（覆盖两个年度周期），不是已找到最优值；本轮不比较其他半衰期。

唯一训练变化为样本时间权重。四类1/2/3/4或5、162特征、NaN/inf处理、目标编码、r80 fitting/S/C拆分、59个月窗口、G1/G4/G2、树深/轮数/随机种子与其他参数均不变。没有类别重加权、重采样、窗口裁短、局部模型、margin锚定、概率阈值调优或G重选。

## 2. 训练与输入

- 复用D34 `geoxgb-d34-e1-brier-20261002`（7b2bf6fe482d0a77a664f3627e493934d976696d）的21组H/T：H={4,8,12}；T={2018-06,2018-10,2019-02,2019-06,2019-10,2020-02,2020-06}。原root只加载，不重训；每组仅新拟合一个加权root，共21次fresh global拟合。
- 原始fitting行顺序与标签保持；`O=T−H`，行标签月为`m_i`，必须在`[O−59,O)`。`u_i=2^(-((O−1)−m_i)/24)`；`w_i=u_i/mean_fit(u)`，均值归一化只在该root的fitting行上计算。以float64计算，转换float32传入XGBoost并记录实际权重；不以S/C/E3数据决定权重或参数。
- 复用`native_xgb.fit_global`的现有训练路径，可加默认`None`的可选sample_weight。无权重调用的数值行为、记录结构必须保持；新分支记录实际传入权重的dtype/hash/sum/min/max/Kish ESS。拒绝长度不符、非有限、非正权重。不要扩展缓存、runner模式或通用训练框架。
- 保持G_CONFIGS中各H配置、booster_params、随机种子与线程配置；加权root从头训练，不是原root追加树。
- 用现有`accept_mode`接收冻结D34、复用`rebuild(..., max_month=2020-12, with_fitting=True)`与原membership/fitting-key哈希守卫。所有分析性Parquet读取限制<=2020-12；不读最终期ledger、不分析2021+标签/分数。
- 只写新D37运行目录，D34/D35及母包不变。顺序执行即可。预计数分钟，未实测，不新增并行基础设施。

## 3. 评分与最小证据

- 首先重放已保存root的同键C/E3预测/概率，必须复现D34；训练加权root、保存并重新加载冻结UBJ后再评分。必要标签比较是重建守卫，不用于选择。
- 主结果：21组E3逐H/T、按H/目标及整体：原root、加权root、persistence。保存所有E3行的四类概率/argmax及危机结果。C仅诊断，亦保存两root的同键概率/预测；不据C选择方案。
- Persistence来自该行精确O的`hist_phase_o00−1`；缺失不回填。全体root对照保留所有行；与persistence比较时三者使用相同非缺失键并报告覆盖。既有dev_baselines覆盖的15组交叉核对；无需新增expert回填或最终期比较。
- 危机主评分仍四类argmax后code>=2；四类macro-F1次指标。另报告危机Brier概率损失（p3+p4或5，仅评分），TP/FP/FN/TN、两root差值。逐折均值差与汇总混淆F1分开，不把21组当独立验证。
- 沿D36固定的00/01/10/11/起点缺失五组报告E3两root的纠正/破坏数与TP/FP变化；它们是事后误差层，不可用于部署路由、国家/区域筛选或调权重。
- 每root保存UBJ、fitting键及标签/实际权重或可逐行重建的键权重表、训练参数、原root来源/hash、窗口/实际月份、行数、权重摘要和producer/runtime身份。直接使用现有身份/哈希工具，不搭新接受框架。
- 事前支持检查：61,361–69,546 fitting行、15–18个实际标签月份；归一化权重约.412–2.040；Kish ESS/N约.805–.821。这里只量化权重集中度，不代表独立样本量或区域支持改善。证据外部`d37_weight_support.json`。

## 4. 检查与执行

先提交本spec及简短PRD/design/implement/上下文指针。原Claude native implement→native check→适用测试→producer提交→冻结Windows Python运行21组→监督方独立核验→结果落盘并综合。

必要检查：旧默认fit_global路径真实同夹具旧/新重放等价；权重公式/仅fitting归一化/传入float32一致，改变C/E3标签不改变权重或训练模型；训练无S/C/E3行、4类及固定G轮数；同键评分和missing-persistence处理。共享fit_global修改前尝试GitNexus impact并列明实际调用者；图不可用时记录源码追踪，不修索引。

主流程/局部训练不启用新权重，默认保持。21组均保留，异常不靠换日期弥补。结果只能回答这一个固定权重是否改善开发E3；即使胜过原root，仍须单独说明是否胜过persistence以及各H/日期的不一致，不能据微小均值增益宣称过拟合已解决或Stage3成功。不自动扩展到分区、其他半衰期或margin锚定。

**证据界限决定（监督方，option 3）**：只做21次加权拟合，不做无权重验证性重拟合。理由：监督方核对原producer `app/main_model_GF.py:448-468,480-482,622`，确认它与`rebuild`使用同一冻结快照、同一有序162列（dtype=float）、同一area/month排序、同一滚动拆分、同一fitting掩码与原生clean；`rebuild`只多了2021前过滤。数据/特征身份、有序成员与标签/fitting键、root重放已守卫此对照，未发现具体的fitting特征差异。可选：不拟合，直接把重建的FIT X/y与按键连接的快照比较。逐字节相同的重拟合证据更强，但对这一有界加权对照并非必需；不声称普遍等价。

## 5. 运行与事实结果（2026-10-02；独立核验与科学综合由监督方负责）

**运行：**
- 生产`3b5398966c3e8d602eeed4b7b3dff2c1fdaf592c`（测试92 OK、exit 0）。
- 运行`C:\Users\swl00\geoxgb_runs\geoxgb-d37-recency-root-20261002`，命令`stage1_recency_root.py --d34-run …geoxgb-d34-e1-brier-20261002 --out … --producer-rev 7b2bf6f`，exit 0，174 s（顺序）。
- 21/21根通过拟合前门槛，恰好21次加权拟合。
- 15个已覆盖折的dev_baselines交叉核对0不一致，2018年6折为`not_covered`。

**gate.json措辞说明：**规则文本“failed roots are not fitted”只适用于拟合前门槛；dev_baselines核对在拟合后执行，失败会阻止summary，但该根文件已写出。本次无此失败。

**默认路径与特征证据：**
- `fit_global`默认路径的实际旧/新重放：f412956的`git archive` vs 新代码，booster字节与记录相同，sha256 `940d83dd…`。证据在`C:\Users\swl00\geoxgb_runs\d37-fitglobal-replay-evidence\`，脚本`research/d37_fit_global_replay.py`。
- 监督方fitting矩阵核对（`d37_fitting_matrix_check.json`，无重拟合）：21个重建FIT矩阵/标签/顺序与独立快照键连接完全相等。

**E3（113508行；原root vs 加权root）：**
- 逐折均值危机F1差−.001361（10正/11负）。
- 汇总F1 .550455 → .548952：TP 9767→9916，FP 5029→5520，FN 10924→10775。
- 四类macro .53186 → .53280。
- 汇总危机Brier损失 .101890546 → .104238011（变差）。
- persistence同非缺失键（112795行）：原.551516、加权.549990、persistence .586406；逐折均值差相对persistence由−.03962变为−.04101。

**按H与目标的E3均值差：**
- 按H：H4 −.008301，H8 +.001280，H12 +.002937。
- 按目标：2018-06 +.008365，2018-10 +.004584，2019-02 −.017795，2019-06 +.004378，2019-10 −.014754，2020-02 +.005924，2020-06 −.000231。

**E3事后五组（加权−原）：**
- 00：纠正177/破坏444，FP +267
- 01：180/88，TP +92
- 10：275/499，FP +224
- 11：360/303，TP +57
- 缺失：0/0

**C（诊断）：**逐折均值差−.007795；汇总F1 .669541 → .662453；汇总Brier .047913908 → .048613557。

**未做：**执行方未运行监督方的`d37_independent_check.py`；无E4、Stage2/3或close；科学解读待监督方综合。

**监督方独立核验（通过）**：冻结Windows Python，未导入生产代码，未重新拟合。
- 21根，210项混淆/精确F1检查。
- 每个模型321047个C/E3概率行。
- 权重float64/实际float32/哈希、fitting键、G、轮数、模型哈希、原始XGB重放全部通过。
- 默认路径旧/新UBJ与记录独立相等；归档的旧`native_xgb.py`等于git f412956。
- 证据保存在`research/d37_independent_check.py`、`research/d37_independent_results.json`、`research/d37_fitting_matrix_check.json`、`research/d37_weight_support.json`（原件在`C:\Users\swl00\geoxgb_runs\`）。

**监督方科学决定：不采用24个月时间加权root。**
- E3全键汇总F1 .550455→.548952，逐折均值差−.001361（10正/11负）。
- persistence同键：原.551516、加权.549990、persistence .586406。
- E3 Brier .101890546→.104238011。
- H4变差，H8/H12小幅改善。
- 净变化+149 TP、+491 FP（稳定非危机00 +267 FP，缓解10 +224 FP）。
- 这不是过拟合已解决。
