# D38 / A12：固定弱 persistence 初始 margin 的 root 对照

状态：2026-10-02监督方批准本轮有限实现与运行。用户已委托一阶段探索与设计；Claude已完整交叉审阅，未发现影响结果的设计缺陷，以下记录采纳的最小补强。必须先提交本spec/上下文，再实现。原任务、审计 run ed632775、base 与 Claude 执行会话保持。无 Stage2/3、完整648、最终期评分或 close。

## 1. 为什么改用这一对照

D37 固定24个月时间权重未改善 E3，不采用。D38 前置零新增 XGBoost 拟合检查已完成：原 fitting 行的 add-one 四类转移表并非 persistence；21根中9根把起点类3的最可能去向判为类2。同非缺失键 E3，转移表 F1 .456666、原root .551516、persistence .586406。该比较是暴露过的开发诊断，不是最终评价。

原始标签日历在2016年前为1/4/7/10月，之后为2/6/10月。原 fitting 行的精确起点标签可用率：H4 .361–.802，H8 .240–.670，H12 .856–.874；E3整体可用率 .993719。此为真实信息可用性差异，不自动等于工程错误或过拟合的因果证明。只保留起点可用行会把H8压缩到4–10个标签月份，本轮不删行。转移表的起点4或5支持也不是空的，不以空单元解释失败。

本轮假设：保留当前数据、特征与有限轮数的树，将四类均匀初始预测改成一个固定、较弱的 persistence 偏好，可能减少无效偏离，同时保留危机发生和恢复两个方向。原root已有 `hist_phase_o00` 等历史特征；本轮检验有限 boosting 的参数化，不声称增加了新信息或初始概率已校准。梯度与 Hessian 改变是该参数化的一部分；不把改进归因于唯一机制。

## 2. 唯一训练变化与冻结范围

- 完全复用 D34 的21组：H={4,8,12}、T={2018-06,2018-10,2019-02,2019-06,2019-10,2020-02,2020-06}，G1/G4/G2、r80、seed42。原root仅加载；恰好21次新鲜 anchored global 拟合。不拟合转移表模型、不重拟合原root、不训练局部模型。
- 59个月 `[O−59,O)`、原始 fitting 行/顺序/标签、162特征、NaN/inf处理、G参数/树深/轮数/线程保持。无样本权重、删行、重采样、类别权重、G重选、半衰期或 margin 强度搜索。
- 每行只用其自身精确起点的 `hist_phase_o00`（合并后的相1/2/3/4或5）；缺失不回填最新标签。fitting行的自身起点为该行标签月减H，不是外层root的共同O。
- 事前固定 `lambda=0.5`。起点类别为k时 `q=0.5*one_hot(k)+0.5*Uniform(4)`：k类 .625、其余 .125。缺失起点 `q=Uniform(4)`。这是单一有限实验选定的弱偏好，不是估计出的最优强度。
- margin 为 `m=0.5+log(q)−mean_class(log(q))`；float64计算、float32传入 XGBoost。缺失起点各类 margin 必须恰为原默认 .5。类序固定 code0..3（1/2/3/4或5），形状 `(n,4)`。无全数据归一化或标签依赖。
- 固定prior-only的缺失行四类概率相等，沿现有argmax规则取code0。报告每根fitting与E3起点可用率；二者的差异可能使原有历史特征作用与margin叠加，按H/T的差异不能被单独解释为预测跨度效应。
- 拟合与每次预测均传入对应行的 margin。保存/重新加载 UBJ 后仍显式传入；不把 UBJ 中的 base_score=.5 误当成已包含逐行 margin。
- 复用 D34 acceptance 与 D37 原root重放/fitting键/窗口守卫。分析性 Parquet读取固定<=2020-12，不读最终 ledger或2021+标签/成绩。只写新的 D38目录，母包与既有运行不变。

## 3. 最小实现与检查

- 复用 `native_xgb.fit_global` 与 `proba`，可以添加默认 `base_margin=None`；保持无 margin 的旧数值与记录完全不变。复用现有 DMatrix 构造，再在需要处 `set_base_margin`；不改主工作流开关或建立新的训练框架。
- 验证 margin 行数/四类轴/有限值与实际float32。anchored booster带一个持久化标记；若用现有预测接口却未提供所需margin，必须明确失败，不能悄悄输出 trees-only 分数。当前 continuation 不支持这一实验，接收到带标记的parent时明确拒绝；不在本轮拓展局部 margin 支持。
- 标记由 `fit_global` 在使用margin时写入booster attribute，先写标记再算模型hash；UBJ重载后保留。`proba`同时拒绝“marked且无margin”和“unmarked却传margin”；不增加可配置rule注册或第二层wrapper。用现有phase映射校验起点值，缺失行直接赋float32 .5。
- 新runner复用既有数据重建、原root重放、基线核对、指标和身份工具；不要复制整套训练/验收框架，不改变D37的结果。
- 必要测试：实际旧/新默认路径同夹具重放一致；公式/轴/float32/缺失中性；改变C/E3标签不改变margin或拟合输入；margin保存后重放概率相同；缺margin预测及未支持的continuation拒绝；正确带margin与不带margin的原始XGB输出差等于 `m−.5`（允许浮点容差），检出转置/漏传。合成夹具拟合不计入21个真实root，不增添真实验证性重拟合。
- 合成夹具补一项中性训练等价检查：全部margin=.5的新模型与无margin模型的树结构和预测一致（模型attribute不同，故不要求UBJ相等）；现有旧/新默认路径仍要求模型字节与记录一致。
- 先尝试 GitNexus impact，失败则记录源码调用者/范围；不修索引。native implement→native check→测试/默认重放→producer提交→冻结 Windows Python3.12.10/XGB3.0.0顺序21组→监督方独立核验。每次提交尝试detect_changes。

## 4. 结果与证据

- E3为主；C仅诊断。原root、anchored root、固定prior-only在全部同键比较；与精确起点persistence比较时四者都取同一非缺失键，并报告覆盖。覆盖的15个dev_baselines折仍须同键/标签/persistence一致；2018年6折明确未覆盖。
- 增加一个无需拟合的固定控制：`p_post = p_original*q / sum_class(p_original*q)`，等价于在原root的输出margin上加同一中心化log q。缺失起点时原概率不变。它与其他模型同键评分，使用同一固定lambda，不做任何阈值搜索。该控制区分训练时参数化的增益与仅预测时调整；若anchored不优于它，不能宣称训练时初始化有额外价值。不要log已舍入为0的概率，也不做epsilon调优。
- 主决策仍四类argmax再折叠 code>=2；危机正类F1，四类macro-F1次指标。报告逐根、按H/日期、整体混淆/F1与危机Brier；逐折均值差和汇总混淆F1分开。prior-only在起点已知键上的argmax必然等于persistence，这是机械核对，不是学习收益。其软概率Brier与one-hot persistence Brier须区分。
- 记录00/01/10/11/缺失的纠正/破坏和TP/FP变化；记录anchored相对prior/persistence的危机偏离率及偏离时谁正确，检验是否只退回persistence。不得依据这些事后组进行路由、筛选或调强度。
- 每根保存UBJ、marker、固定margin规则与实际常量、原root来源/hash、G参数/轮数/fit记录、原fitting键及标签/实际float32 margin（或可逐行重建的起点code表与margin hash）、各集合的四类概率/argmax/真值/persistence、producer/runtime身份。复用原工具，证据留在独立D38目录。
- 21组均保留；不因结果差换日期。训练遇到错误先诊断，不自动扩展拟合预算。即使优于原root，仍分别说明是否优于persistence、是否保留发生/恢复能力及各H/日期异质性。开发集反复暴露，无显著性/最终泛化承诺。
- 事前解释规则：若只有F1上升而危机Brier变差，只报告决策折衷，不以此认定过拟合减轻或概率预测能力改善，不直接推广到分区。与prior-only相比必须存在学习带来的收益；未超过persistence仍是未达目标。单一F1或Brier改善都不是Stage3成功。
- prior-only在已知起点键的危机F1与persistence相同，故不是另一个独立F1门槛；二者概率损失的不同仍独立报告。
- 本轮结束后停止新训练，由监督方综合；不自动扩展margin强度、局部树、Stage2/3或最终期。Stage1过拟合仍需其自身证据，root改善不等于分区问题解决。

## 5. 运行与事实结果（2026-10-02；独立核验与科学综合由监督方负责）

**生产与运行：**
- 生产提交2d4fe4e3dac5bff68327c5a426d72f41c300c212；提交后`tests/test_baseline.py` 100项OK，exit 0。native implement约5.9分钟，native check约4.3分钟，无影响结果的发现；实际旧/新默认路径重放字节一致（booster sha256 `940d83dd…`，证据`C:\Users\swl00\geoxgb_runs\d38-fitglobal-replay-evidence\`）。GitNexus impact/detect_changes均为LadybugDB只读错误，风险UNKNOWN，调用者按源码追踪，全部走默认路径。
- 运行`C:\Users\swl00\geoxgb_runs\geoxgb-d38-persistence-margin-root-20261002`（`stage1_persistence_margin_root.py --d34-run …geoxgb-d34-e1-brier-20261002 --out … --producer-rev 7b2bf6f`，冻结Windows Python），exit 0，244 s；21/21预拟合门通过，恰好21次anchored拟合，21个UBJ均带标记`d38-persistence-lambda0.5-v1`；dev_baselines 15折核对/6折未覆盖，0不一致（日志`C:\Users\swl00\geoxgb_runs\d38-run.log`）。
- fitting起点可用率：H4 .361–.802，H8 .240–.670，H12 .856–.874；E3每根≥.974。

**E3全键汇总（113508行；TP/FP/FN，危机F1，四类macro-F1，行加权危机Brier由保存行推导）：**
- 原root 9767/5029/10924，.550455，.531864，.101891
- anchored 10398/5973/10293，.561114，.545089，.101314
- prior-only 11707/7614/8984，.585174，.587866，.136422
- post-hoc 11517/7305/9174，.582947，.583836，.115894

**persistence同键（112795行）危机F1/Brier：**原.551516/.101885；anchored .562155/.101291；prior-only .586406/.135704；post-hoc .583974/.115976；persistence .586406（one-hot Brier .146407）。post-hoc与监督方预先计算的固定控制一致。

**逐折（21根）：**
- anchored−原 危机F1均值+.012553（12正/9负），Brier均值−.000578；anchored−post-hoc −.023213（7正/14负）。
- 按H anchored−原 F1：H4 +.007235、H8 −.005522、H12 +.035946；逐折均值Brier原→anchored：H4 .090806→.089900、H8 .110826→.109448、H12 .104161→.104710。

**事后组（E3同键，相对原root）：**
- anchored：00 FP −338（纠正530/破坏192）；01 TP −213；10 FP +1282；11 TP +844。
- post-hoc：00 FP −1469；01 TP −647；10 FP +3745；11 TP +2397。
- 与persistence不一致的行：原9103（模型对4871），anchored 6426（3376），post-hoc 845（479）。

**C（诊断）：**汇总F1原.669541→anchored .668291；行加权Brier .047914→.048049。

**未做：**无局部margin拟合、λ搜索、Stage2/3或close；科学解读待监督方综合。

**监督方独立核验（通过）**：无生产导入、无重拟合。
- 21根，336项混淆/精确F1与Brier检查；每个模型（原/anchored/prior-only/post-hoc）321047个C/E3概率行。
- 实际拟合float32 margin/hash、键、标签、G、轮数、标记、模型hash、原始XGB概率重放与偏离计数全部通过。
- 透明记录：核验脚本首次失败是监督方自身的XGBoost缓存误用（先无margin预测，再改同一DMatrix的base_margin后再次预测，XGBoost沿用缓存预测）；改用新DMatrix即通过。生产`nx.proba`每次新建DMatrix，非产出结果问题，无需修复产品。
- 执行方精确比对监督方预先计算的固定控制（`d38_fixed_controls.json`）与本次运行E3行：21根×原/prior-only/post-hoc共63项，计数、精确F1、Brier全部相等；整体全键/同键/按H计数与F1相等，Brier差≤2.8e-17（求和顺序）。
- 证据保存在`research/d38_independent_check.py`、`research/d38_independent_results.json`、`research/d38_fixed_controls.py`、`research/d38_fixed_controls.json`、`research/d38_fixed_controls_match.py`、`research/d38_fixed_controls_match.json`（原件在`C:\Users\swl00\geoxgb_runs\`；JSON换行已规范化）。
- 训练已停止；科学综合待监督方。
