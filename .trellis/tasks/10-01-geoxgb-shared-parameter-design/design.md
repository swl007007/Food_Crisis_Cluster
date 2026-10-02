# GeoXGBoost 技术设计 v1.0

**当前覆盖修订：D26–D31已完成；D32/A7已完成（c079b75）；D33/A8已完成（69f2cc3）；D34/A9 E1配对对照已批准实现（d34-e1-brier-contrast-plan.md）。只改变搜索S的历史范围，Stage1过拟合尚未解决。**

**首轮设计按D24采用，D25已授权开始执行；实际状态以审计start和task.json为准。** 本文记录架构/边界/证据；数值预算见experiment-plan.md，执行顺序见implement.md。
母包按D1确认为 `FEWSNETFourClassBaseline/`，标签为 `1/2/3/4或5`；D4冻结共享树并只追加局部增量。v1.0为决策收敛整理，不改变已采用的实验方案。
D5 已确认保留三阶段、重学 partition 和 consensus 地图，并保留旧 RF 地图对照。
D8 确认评价分层，D10 接受 E1/E2 复用及其偏差边界，D20 后明确 Stage 1 区域内随机验证，取代其原内部时间方案。D23 将 D12 扩展为 Stage 1 E2 严格增益 >0 与 >0.01 两族；D13 的 Stage 3 每区相对 global 严格增益 >0.01 保持不变。逐层契约见 `evaluation-contract.md`。D9 已修订为固定59个月标签窗，D11 接受局部缺少“4或5”类。

## 已确认的信息与特征边界（D6/D7）

Expert 仅用于比较；其预测及派生值不进入特征、预测纠偏或融合。不得把 expert-assisted 模型作为本任务独立 GeoXGBoost 的替代。
现有 exact-origin 真实 IPC 及合法历史仍可作为输入；训练行也须按各自 origin 构造，不能使用目标时点真实值。
Expert 在 H=4/8 的合法同键 cohort 上比较；H=12 不制造 expert proxy。
D7 固定既有 162 列特征的来源、定义、有序 schema 和 origin 对齐；保留全部历史特征，不新增/删除字段或改变工程公式。
母包的特征来源为 `.trellis/tasks/archive/2026-09/09-28-fewsnet-four-class-perturbation/feature-contract.md` 及对应有序 schema，具体schema哈希记录于implement.md；实施前复核。
缺失值模型处理已按 D14 固定：新 XGB 保留 NaN，使用原生缺失路由；统一保留 ±inf→NaN 清洗，不采用 RF max_plus 填补，不改动冻结特征公式与真实零值。详细输入契约及源码依据见下文。

## 已确认的科学成功范围

D2：主候选 GeoXGBoost 必须在 H=4、8、12 各自固定的同口径评价键上，fixed-four macro-F1 均严格超过 persistence。
任一 H 未超过即未达整体成功要求；总体平均提升仅可辅助描述，不替代逐 H 判定。
D3：各 H 的配对 macro-F1 增益 95% 置信区间下界还须严格 >0；暂不另设 +0.01 等最低增益。
最终区间按 D17 使用国家块配对 bootstrap：整国的区域和完整时间序列一起抽样，双方共用键与国家权重，在每次抽样的合并 confusion counts 上重算 macro-F1 差值。沿母包 seed=42、2,000次有效抽样、最多20,000次尝试和2.5%/97.5% linear分位数；完整契约见 `evaluation-contract.md`。解释限于已观察历史与保存的预测，不涵盖未知未来时间冲击，也不把三个边际区间称为同时95%。
不得把这项科学目标当作运行后继续针对最终评价调参的授权。

D21/D24将超过expert保留为H4/H8合法同键cohort上的明确优化目标，报告点增益及区间；首轮不增设与persistence相同的expert硬CI验收。

## 问题拆分

四分类已有 partitioned RF < pooled RF < persistence/expert 两层落差。
限制局部复杂度只针对第一层；超过 persistence 还需要全局预测能力改善。
通过 pooled 和固定旧地图对照，分开评估 backend、局部共享和主线重学分区的贡献。

## 探索性实验路线（D22–D24）

研究目标是找到预测有效的方案；候选数量、分区差异和训练分数都不是最终目标。固定162列特征、59个月、四类、信息截止及比较键；按D24的有限方案依次筛选全局、比较新图完整流程，再对所选配置运行旧图诊断。

1. **全局底座能否胜过 baseline？** 先比较少量 pooled XGB 容量配置。若全局本身明显落后，需要先理解底座误差；增加分区不能被默认视作解决办法。此步骤是诊断顺序，不新增“pooled 必须先胜出才准试局部”的硬门槛。
2. **共享是否减少局部模型损失？** 固定同一旧 RF 地图，对比 pooled、独立 local、冻结全局＋局部增量；保持开发预算／评价键可比。检查增量实际改善哪些时期／区域、何处触发回退，以及缺稀有类时的表现。随后主线仍按 D5 重学地图，不以固定图实验替代它。
3. **较宽松的候选生成是否有益？** D23 已批准 Stage 1 E2 严格 gain >0 与 >0.01 两族；两者均使用已定 80/20、50/50 随机划分、固定共享机制，父模型赢平局。E3 外部目标月评分及 E4 正权重规则保持，Stage 3 继续 >0.01。较松门槛可能保留有用结构，也可能增加噪声；须经开发期完整预测流程评价，不能从候选数量增加推导有效。
4. **候选池是否过度集中？** 固定少量 seeds，记录候选数量、重复结构和权重集中；结合三种门槛组池策略的开发预测表现解释。相同或相似候选可能说明稳定，也可能说明共同偏差；不强求低相似度，不按最终成绩追加种子。本轮不做seed数量扫描，不能据此声称已测得候选收益饱和点。

第三项由D23扩展D12，Stage 3的D13不改变。开发期有界探索不改变D16/D18：最终地图、参数及更新规则须冻结，最终评价不反向用于择优。实现正确与D3科学胜出分别报告。D24已确认 `experiment-plan.md` 的预算、参数表和选择顺序；早先五seeds／270候选建议已被取代。

## 三阶段主线与比较设计

**D19/D20范围修订**：Stage 1区域内随机80/20与50/50取代此前内部rolling-origin方案；Stage 3时间验证与D18条件历史筛选保持。D24固定seed数和候选预算，不把两种比例误写为仅两张图。

D5 的主线：Stage 1 用 D4 共享机制的 XGBoost 学习候选分区；Stage 2 沿用既有 consensus 方法生成新地图；
Stage 3 在合法滚动历史上拟合该 fold 的全局树及新地图区域增量。新图分区数由既有 consensus 规则决定。

保持同一数据、时间、特征和评价契约：

1. 原 GeoRF、persistence、合法 H 的 expert：比较锚点。
2. pooled XGBoost：只换全局底座。
3. 固定同一地图、各区独立 XGBoost：识别共享机制贡献的对照。
4. 同一旧 RF 地图、共享全局树 + 小型局部增量：固定地图对照（D5）。
5. 共享 XGBoost 重学 partition、沿用 consensus 得到新地图，再拟合全局共享树＋局部增量：主候选（D5）。

主线、旧图独立/共享local诊断及运行预算均已按D24确认。旧图两臂使用主方案所选G/L，不额外搜索，不参与主方案选择。
只改变 Stage 3 的固定地图实验不能代替主线；不得在最终评价上择优选择旧图或新图来重新定义主候选。
新 XGBoost 各臂拟合窗口均固定 59 个月；既有 RF v7 的 35 个月结果只作历史参考，不把两者差值解释为单独更换 backend 的受控效应。若需同窗 RF 重跑，另在最终实验预算中明确，不由本次修订自动增加。

### Stage 1 候选族（D20 已确认比例与随机方式）

在每个开发目标 T、H 对应的 O=T−H，先固定 `[O−59,O)` 的合法历史池，再按 80% train /20% validation、50% train /50% validation 区域内随机划分。使用固定 seed 列表，不反复尝试直到出现正权重；每个候选 fitting 与 validation 键互斥，父／子只拟合 fitting 行，E1/E2 复用 validation。随机验证不解释为各行原始起点的前瞻效果。E2 按 D23 保留严格 >0 与 >0.01 两个 threshold_family，Stage 3 门槛不变。

候选质量另用未进入该候选fitting/E1/E2行池的目标月T作E3评分；两臂同键同fitting规则，E4沿用既有权重。比例、seed、容量及无支持处理按D24冻结。固定每H的L和一个门槛族、每比例一个seed时为3 H ×9目标月 ×2比例 =54候选；两门槛为108，三seeds为324。数量仍须结合正权重、重复与集中诊断，不表示独立多样性。D24先全开发期选G，E3因此有配置选择偏差，见experiment-plan.md第2节。

D20保留Stage 1候选行池的外部origin截止，放开内部随机验证的逐origin拟合要求；Stage 3内部预测仍按自身V拟合。D24另允许全开发期选择G，须披露该配置选择偏差，最终Stage 3标签/成绩不参与候选方案选择。复用母包 `src/utils/split.py:9-95`：区域内ceil(n×val_ratio)，至少一条验证并保留训练，singleton留在训练；split seeds为42/43/44。

现有 Stage 2 是 **fs1–fs3 候选合并形成一张 general consensus 地图**，不是每个 H 各自一张图（母包 `scripts/run_stage2.py:68-79`）。沿用 D5 时，应记录候选的 H／方案身份并共同进入该池；不得因每 H 调参自动改成三张地图。开发外层构图的候选时间截止对池中所有 H 都适用。

已确认候选预算见experiment-plan.md：三seeds×两比例×两门槛；一套跨H局部配置的合并池324个候选，开发两套L共648任务，24组方案复用这些产物。数量不等于独立信息量或内部estimator拟合次数。

旧 v7 仅 3/27 候选有正权重、其中一个占 51.72%，见 `research/candidate-diversity.md`。新池需报告未分裂／分裂、正权重、同覆盖下标签重命名不变的分区重复及权重集中；不同覆盖单独披露。增加数量不保证增加有效结构差异，诊断不自动删除重复／零权重候选或改变 E4。本次有限方案不增加删 seed 重跑，先比较已提出的三个组池策略。

D21 的取舍原则：候选趋同可体现稳定结构，也可能来自共同偏差；不为追求差异删除一致候选或反复抽 seed。D23 的门槛对照是明确研究假设，不是为凑分区数临时降门槛。Stage 2 可生成不同于任何单候选的新图，不保证预测最优。多样性／稳定性仅作诊断，有限方案按开发预测表现比较，最终评价不参与择优。

## 已确认的训练窗（D9 修订）

实际标签窗统一为 `[origin-59,origin)`，不做35/59择窗；各H/全局/父/局部采用同一规则。内部rolling-origin模型按自身起点完整回溯，不限于当前外部fold历史池。时间对齐和已核验的源可用性限制保持，不冒充已核验真实发布日期；早期支持、保留NaN及回退按experiment-plan.md第3/5节。
59 个月指标签筛选窗口，不是 59 次真实标签，也不把既有 162 列特征的工程回溯长度一并改成 59。35/71 个月的计数保留为研究记录，不进入新窗口搜索。
当前母包 `scripts/prepare_fourclass.py:65-70` 的 `SNAPSHOT_FIRST_MONTH=2014-01` 对应旧 35 个月池，不能直接覆盖新窗口。例如 Stage 1 T=2018-02、H=12 的 O=2017-02，新 fitting 标签下界为 2012-03；内部验证 origin 还可能需要更早快照。新 fork 必须由冻结原始源和原特征公式按实际 origin 日程重建快照，核查源数据覆盖；不能只改 TRAIN_WINDOW 后沿用被 2014 年截断的缓存。

## 已确认的调参范围（D15）

每个 H 在开发期统一选择一套全局配置和一套局部增量配置，二者可以不同。选定后跨分区、Stage 1 分割层以及最终滚动 folds 复用，不逐区／逐节点另搜深度、轮数、学习率和正则。不同区域仍以自身合法 fitting 行学习增量树，D12/D13 只选择这套受限增量或零增量，不把回退变成区域专属调参。
共享超参数不能替代D4冻结并继承父/全局树；也不表示各区域拟合出同样的新树。G/L参数、固定轮数及80-round路径上限统一见experiment-plan.md第2/3节。无early stopping，不通过逐区选择轮数绕开共享配置。

## 已确认的时间边界（D16）

按 D16，2018–2020 目标月用于开发与地图学习，最终参数／地图选择只用截至 2020-12 已可用信息；随后冻结配置、候选地图选择及自动路由规则。最终按 H4=2021-05、H8=2021-09、H12=2022-01 起至 2024-12 的原日程滚动评价，只对真实存在的标签评分，空月份不补造。来源为 `scripts/prepare_fourclass.py:67-70`；实际观察月份较稀疏，见研究计数。
后续fold已到达标签可按59个月规则更新模型及D13路由，不能据最终报表重选参数/阈值/地图；冻结的是规则，不是2020年的模型。旧2021–2024基线已查看，结果称回溯评价。D24日程见experiment-plan.md第5节：开发地图只消费评分目标早于O的候选；先全开发期选G使配置并非历史O已知，须披露选择偏差。此取舍已采用，最终2020-12截止不变。

## 已确认的部分共享机制（D4）

每个 horizon / 时间 fold 单独拟合全局树集合，不把未来 fold 模型传回过去。

    F_g(x) = F_shared(x) + Delta_g(x)

同 fold 内 F_shared 的已学得树在各区固定相同；Delta_g 仅新增有限、浅层、强正则的树。
四分类按固定类别轴使用softmax；容量按D24有限配置和路径上限。
共享树的结构和叶值均保持不变，不能通过局部 refresh 改写后仍声称共享相同参数。
输入特征顺序和变换须兼容共享模型；按 D14，全局、父、子使用同样的 dense 数值输入、NaN 缺失约定与无参数 ±inf→NaN 清洗，不再由子分区拟合 imputer。共享前缀的既有缺失分支方向随树结构、叶值一并冻结。

partition 学习期间对应为：

    F_child(x) = F_parent(x) + Delta_child(x)

子分区继承父树，仅新增有限容量；允许 Delta_child=0，继续用父模型。
按 D5 保留三阶段 consensus，Stage 1 父子树只用于学习地图；Stage 3 在该 fold 合法历史上重拟合全局树和区域增量。
不能把不同月份、不同训练身份的 Stage 1 booster 直接拼接成 Stage 3 模型。

采用原生 `xgb_model` continuation，执行前验证版本兼容。默认 process_type 创建新树；update/refresh 修改旧树是另一机制，
修改后各区叶值不能宣称仍是同一组共享学得参数。复制相同全局树前缀属于统计参数共享，不保证物理存储去重。
父模型过深时，局部增量仍会过拟合，因此全局和局部容量须共同控制。

## 已采用的过拟合控制及限制

- 明确区分 q 搜索、分割接纳、Stage 1 留出/consensus、开发选择和最终评价。
  D20 的 Stage 1 q／接纳复用区域内随机验证，训练／验证键互斥并受候选外部 O 截止约束；不要求每条随机验证行的训练截止早于该行自身 origin。Stage 3 每个内部预测另按自身 V 限制训练标签。
  D10 的复用和选择偏差边界保留，Stage 1 原多起点时间方案由 D20 取代；内部接纳不称独立最终测试或显著性检验。
  Stage 1 目标月评分参与 consensus 地图构建，属于开发证据，不是最终科学检验；D3 的 95% CI 门槛仅用于最终主候选与 persistence 的逐 H 比较。
- 调参范围按D15，具体G/L网格、固定轮数及路径上限按D24；只在固定候选中选择，不独立调区、不加early stopping。
- min_child_weight 是 Hessian 支持约束，不等同真实行数或各类支持量；后两者需单独纳入局部资格判断。
- 小样本或内部收益不足则回退父/global；Stage 1按D12/D23，Stage 3按D13，支持资格按experiment-plan.md第3节。D11不因单独缺“4或5”禁用，不要求四类齐全；未观察类别不声称局部效果已验证。支持底线不是统计充分性保证。
- 分区反复搜索同一 validation 的偏差不能靠 early stopping 自动解决；候选搜索与外层时间评价应分开。
- 反复在同一验证集选择回退阈值也会过拟合；固定搜索预算、候选集、主指标与停止规则。
- 按 D5 沿用既有 consensus 方法，由新的候选分区证据重建地图；旧 RF 地图作为固定对照，区分模型和地图的贡献。
- 未胜出须报告失败；不得反复查看最终评价后继续修改方案。

## 已确认的 Stage 1 接纳（D12/D23）

E2 比较当前父模型与 parent/parent、child/parent、parent/child、child/child 四种路由。所有组合使用该候选完整且相同的父随机验证键，汇总两侧 confusion 后计算 fixed-four macro-F1；父／子只拟合 fitting 行，前缀可追溯。按 D23 的候选 threshold_family，最优组合必须严格超过 0 或 0.01，等于门槛拒绝，父模型赢平局。E2 仅作内部选择，不作用于 q、不替代 Stage 3 多起点 gate 或 D3 最终科学验收。

## 已确认的 Stage 3 启用（D13）

当前母包 `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:155-173` 在满足样本／类别资格后直接用 local，没有性能门槛。新设计按 D13，对每个分区使用本次外部 fold origin 前已可用的历史时间验证键，配对比较该区的 global 与 global+local；合并各内部 origin 的 confusion counts 后计算 fixed-four macro-F1。只有严格增益 >0.01 才启用增量；等于 0.01、更低或没有可用验证收益证据时零增量回退全局，不使用本次待预测目标标签选路由。
比较双方使用各内部起点的合法模型，不能拿当前外部global回看过去。通过的路由应用于本次合法历史重新拟合的模型，不直接复用内部booster。保存验证键、模型身份、分数/差值/路由；D24固定最近六日期及资格，局部拟合失败日期仍保留配对行，当前拟合失败仍回退global。
D13 使用该分区的验证总体，与 D12 完整父分支总体不同；相同数值门槛不合并其数据角色。它是内部路由选择，不是显著性结论，也不保证逐区胜出必然提高整体 macro-F1。D11 缺稀有类本身不禁用继续适用，整体仍需外层评价与 D3 验收。

### 已确认：地图在历史筛选中的信息截止（D18）

设本次外部目标为 T、origin 为 O=T−H，内部验证目标为 U<O、内部 origin 为 V=U−H。每个内部模型仍只拟合 `[V−59,V)` 的合法标签，不能使用当前 O 的模型回头预测 U。用户已选择以下条件历史筛选口径：

- **条件历史筛选口径（D18 已批准）**：使用在本次 O 已可用的地图，对 U<O 的历史进行 global/local 配对筛选；不要求这张地图在更早 V 已存在。每个内部 booster 的拟合截止仍为 V，但地图可能已利用 U 或更晚、仍早于 O 的标签，因此这不是完整策略在 V 的前瞻重放，存在地图选择偏差。筛选分数不能当独立有效性证据。
- **未采用的严格内部时点口径**：同一地图也必须在 V 已可用。对于信息截止为 2020-12 的最终地图，保守要求 V>2020-12。实际有标签的外部 folds 中，H4/H8/H12 首次有一个合格内部验证月分别为 T=2022-02/2022-10/2023-06；首次有三个分别为 2022-10/2023-06/2024-02。此处保留为取舍证据，三个不是已批准的支持门槛。

条件口径保留较早验证月份：开发地图只消费评分目标早于外部O的候选；G按D24在全开发期选出，开发目标可影响该配置选择，故开发不是严格独立外层检验。最终外层按D16冻结地图/参数/阈值，当前目标T标签不得进入任何本次选择、拟合或路由。到达标签仍可进入后续合法fold的拟合/固定gate，禁止据最终报表改规则。D18的条件地图偏差、D10的E1/E2复用偏差、D24的全局配置选择偏差分别披露。

最近六个真实日期原为计数探针，现已由D24采用；至少三个日期同时有本区真实验证行和合格局部拟合才满足gate日期资格。每个最终fold有六个全局日期不等于每区都合格；见 `research/sample-support.md`、`calendar-support-counts.json` 和experiment-plan.md。

## 已确认的缺失处理（D14）

新 XGB 路径按 D14 保留冻结特征工程的 NaN，采用树模型原生缺失路由，取消模型侧 MaxPlusImputer；保留原有无参数的 ±inf→NaN 清洗，不把真实零或其他有限值改成缺失，不新增／删除缺失指示列。全局、父、子和预测路径使用同样的 dense 输入表示及 `missing=NaN`。共享树前缀包括其缺失值分支方向，局部追加不能改写已有树的该方向。该选择避免子区重新拟合填充值改变共享树输入，但不同于 RF 原有插补，不能预先断言性能提升；原 RF 包与结果保持独立。

当前母包事实（路径均相对 `FEWSNETFourClassBaseline/`）：`scripts/prepare_fourclass.py:287-304` 将 covariates/history 按 schema 拼成快照，`app/main_model_GF.py:80-99` 直接转换成 dense float 数组；`src/feature/fourclass_features.py:73-108` 的精确月查找及 trailing_sum 保留／传播 NaN，历史计数为零和已有 absent indicators 是特征定义，不是可随意替换的 missing。NaN 尚未在活跃入口的 adapter 之前统一填掉。
`src/model/model_RF.py:30-69,154-159` 的 MaxPlusImputer 在各 estimator 实际 fitting 行上学习 max×100 填值（max=0 时100，全缺失列0），并在清洗时把 ±inf 转 NaN；Stage 3 `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:72-97` 的 FittedRF 采用同样流程。当前独立 child 可以重拟合 imputer，因为它替换整个森林；这不能直接迁移到共享父树前缀的 XGB continuation。
官方依据：2026-10-01 经 Context7 查询 [XGBoost FAQ](https://github.com/dmlc/xgboost/blob/master/doc/faq.rst) 与 [Python intro](https://github.com/dmlc/xgboost/blob/master/doc/python/python_intro.rst)，树算法学习缺失的分支方向，missing 默认 NaN；稀疏矩阵缺项与 dense 的真实零含义不同。本轮只核查文档和源码，没有拟合 XGB 或验证本地版本行为，也未追溯外部源 panel 在生成前是否曾填补。

## 样本支持对设计的约束

用户要求先查样本是否足够。`research/sample-support.md` 记录 v7 真实键、候选分支汇总、13 区旧图及方法资料；`research/sample-support-counts.json` 保存计数与输入 hash，配套脚本只计数、不训练。
Stage 1 行政区历史标签中位数 8，现有 q 验证标签中位数 2；这不等于局部模型只拟合两条，候选分支会合并多个行政区。Stage 3 各区 fitting 行数中位数 2,805，但仅 8 个真实月份，40% 的分区×fold 缺“4或5”类。不能把 ≥50 行或 162 列的行列比当充分性证据。

两个相邻 6 个月验证块实际只有 2／1 个观测月份；若父树拟合也硬限制在当前 35 个月池内，H12 的最早验证前缀只剩 2 个月份。这是旧窗口的计数诊断，D9 已改为 59 个月，Stage 1 内部方案又由 D20 改为随机验证。Stage 3 每条内部预测的训练标签仍按其 origin 检查，不能把旧 Stage 1 时间探针误当最终日程。

W=59 的真实标签窗在 Stage 3 可覆盖 14 个观测月份，但仍有 35.4% 分区×fold 缺“4或5”；长历史可能增加漂移。用户已修订 D9 为直接固定 59 个月，并按 D11 接受稀有类缺失这一现实。继续记录支持，不因该类缺失本身禁止增量模型；无支持类别的局部效果仍不可确认，不默认新增逐节点显著性检验。

已确认的证据取舍（D10/D20）：E1 搜索与 E2 接纳复用随机验证行，仅作候选外部 O 截止下的内部选择。边界／路由选择看过这些验证标签，不能称独立接纳检验。整个流程仍须外层时间评价；E3 参与 consensus，不自动替代最终科学证据。旧多 origin 研究保留为历史证据，不再要求用于 Stage 1。

标准四分类one_output_per_tree、num_parallel_tree=1时，r rounds对应4r棵树。experiment-plan.md同时声明rounds、树数及深度；min_child_weight为Hessian总量，真实行/日期/类别另计。尚不能证明门槛足以泛化或共享增量已优于pooled。

## 暂不优先

完整深局部模型外再加复杂 gate/ensemble；自定义 joint multi-task booster；直接切旧 XGB adapter；
同时引入 ordinal loss、大量新特征和全新聚类算法。仅在明确开发证据支持时再讨论。

## 本轮证据与技术待核查

主代理完整读取的既有设计（作为背景，不自动继承为新设计批准）：

- `.trellis/tasks/archive/2026-09/09-28-fewsnet-four-class-perturbation/design.md`
- `.trellis/tasks/archive/2026-09/09-20-fewsnet-clean-persistence-baseline/design.md`
- `FEWSNETFourClassBaseline/README.md`
- `evaluation-contract.md`：当前四分类调用链的逐层核验、q 公式及数据角色。

旧实现：`src/model/model_XGB.py:133-140,234-268` 每次新建并 fit；
`src/partition/transformation.py:733-757` 父模型回退限定 RF/DT。

官方资料，2026-10-01 经 Context7 查询，属于当前 upstream 文档，尚非本地版本兼容验证：

- https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst — process_type default/update。
- https://github.com/dmlc/xgboost/blob/master/doc/treemethod.rst — refresh 更新叶值并保留结构。
- https://github.com/dmlc/xgboost/blob/master/doc/tutorials/kubernetes.rst — xgb_model 恢复训练。

后续有界验证：固定类别轴及局部缺类的 continuation、全局树前缀不变、父模型不被原地改写、
新增轮数的含义、JSON/UBJ保存重放、D14缺失路由及真实零约定在固定XGBoost3.0.0中的兼容性；无early stopping。
本轮没有训练模型，没有声称共享机制已实现。

## Fork 和后续阶段

已选定 `FEWSNETFourClassBaseline/`；source commit/release、独立目录 `FEWSNETGeoXGBExperiment/`、旧结果身份与隔离import边界见 `implement.md`。
只复制55个已跟踪源／包文件，不复制大批 runs/，不覆盖旧结果；当前未创建 fork。
最终规划及真实上下文清单已完成复核，D25已授权执行。先按AGENTS.md提交规划、绑定实际Claude会话并审计start；设计批准D24本身不启动执行。

## D26 修订（2026-10-01）

保留原生四分类 XGB 概率与固定类轴，评价终点改为二分类危机（IPC≥3，argmax 后折叠），主指标 crisis-positive F1；Stage 1 的 E1–E4 改为该终点，先在正确几何的有界子集验证。数值预算与边界不变。详见 experiment-plan.md A1。

## D27：有界时间块对照，问题仍未解决

D26 已观察到 31 个分裂候选全部改善 E2，但其中21个 E3 退化；这是有限开发证据，不证明分区思路无效，也不证明负增益具有统计显著性。D27 只检验区域内随机验证的时间混合是否造成乐观筛选。

在当前候选合法59个月池中，按全池真实标签月份排序，最近三个完整月份为共同 E1/E2，其余更早月份为 fitting；分支继承同一月份角色，不逐区补选日期或把 singleton 验证行移回 fitting。该划分不同于按每条验证行自身预测起点重拟合，不能称严格内部前瞻重放。E3 目标保持排除在历史池之外。

复用现有 split/driver/诊断入口，增加明确的时间块身份及六候选预期集合，不能冒用 r80 身份或默认展开 L2/gt001。保存同键预测，并与 D26 同 H/T/G/seed 的 r80/L1/gt0 比较。时间块改变 root 的实际 fitting 行，因此同时报告 root 的 E3 变化和 local 相对各自 root 的增益，不把整个差值归因于分区。

共享树、L1、80-round路径上限、支持底线、几何及二分类终点保持。小区域支持不足仍按现有规则继承父模型，不新增危机正例门槛。不同 split 的模型不得互相复用；旧 D26 产物保持不变，新代码/运行身份独立记录。详见 experiment-plan.md A2 与 evaluation-contract.md D27。

## D28：共享root的单次局部增量（A3实施与运行已批准）

原Stage1：`F_child = F_parent + Delta_child`；新臂：`F_child = F_root + Delta_child`。每次新子模型都从本候选不可变root独立加载，以该子区原fitting行追加一次L1（20 rounds）；不把祖先增量作为初值。root、兄弟及当前父模型保持不变。空间可以继续细分，终端模型实际只有0或20个root后新增rounds；拒绝新模型时继承的祖先模型也只含一次增量。

拟合来源与接纳对象不同：E1误差仍由当前父模型产生，E2仍比较当前父的四种路由组合，不能将更深层接纳基准改成root。未选子侧保留当前父模型；只有其父本身是root时才回root。

现有 `transformation.py:752-754` 以模型累计轮数限制下一次拟合资格；只重置前缀会同时放宽搜索机会。因此分开记录实际模型增量和路径搜索额度：root额度为0，接受一个新局部模型后继承父额度并加20，沿路径最多80；继承父路由时额度不变。原五个二分层级及支持规则不变。搜索额度不是实际树数，也不保证两种机制产生完全相同的树形或拟合次数。

复用现有adapter、分裂入口和有界比较逻辑，加明确 `root` / `parent` 增量来源模式及最少元数据，不新建模型框架。对照回到D26的r80/L1/gt0，使fitting行、G、root预测保持匹配；D27时间块只作已完成背景，不与本轮同时改变。详见experiment-plan.md A3。

## D29：冻结候选后的确认诊断（已完成，待科学评审）

保留D28的r80 fitting键、G/root、root单次L1、路径搜索额度和父路由语义。仅将原validation以预定无标签规则分成S/C；E1/q及E2仍复用S，C不参与候选生成。E2会改变后续搜索，所以本轮不把“E1/E2各用一半、继续递归”误称独立接纳；隔离的是完整搜索与新增确认C。

搜索入口只能获得fitting/S的监督信息；C不得参与scan、局部支持资格、E2、停止、重试或参数/路由选择。整个候选的map、booster和路由冻结后，以既有predict-only路由分别给C和E3目标行出预测，不在评价前合并S/C重训。C中没有搜索支持的区域沿用既有area fallback，全部评价键保留，不用C标签补分支。原完整几何、fitting区域、NaN/特征契约不变。

C首轮只诊断，没有C门槛，不删图/剪枝/回退，不消费C得分计算E4；E4仍由E3得分生成。记录现有候选产物的冻结身份与C逐键预测即可，不新建通用gate/完整性框架。A4固定分割、六根预算、D28同root主对照及解释限制；原parent/tb3/rootinc臂保持可复现。

## D30：搜索历史与拟合历史分开

在D29 rootconf路径复用S/C拆分和冻结评价；从原validation日期预定最近六个观察月份，只允许其中原S进入搜索。更早S记unused_search_history，root及child仍使用原59个月fitting。C及unused监督信息不进入搜索支持/接纳/停止，保持原fitting地理覆盖。改动只表达明确的近期搜索模式，复用既有split/driver/compare。细节及日期表以d30-recent-search-plan.md为准。

## D31：逐区等量搜索抽样（已完成）

在D30路径上增加显式搜索抽样模式：重建D29 S/C后按A5日期得到逐区k_a，每候选新建`random.Random(search_seed)`，按数值区序对每区日期排序的原S键调用一次shuffle并取前k_a，其余为unused_search_history。search seed与split/confirmation seed分开记录于身份与root元数据；fitting/root/C/E3/几何不变，报告器扩展现有比较脚本并以固定producer接受D29/D30。精确规则、预检及报告项以[d31-matched-search-plan.md](d31-matched-search-plan.md)为准。

## D32：预测路由与空间证据分开

`correspondence_table.csv`为兼容的预测路由导出；新增`assignment_evidence.csv`为Stage1空间证据权威（无搜索行区`spatial_partition_id=s-1`）。见[d32-stage1-assignment-plan.md](d32-stage1-assignment-plan.md)。

## D33：截断路由

depth1用截断`s_branch`（`""`/`"0"`/`"1"`）与最终`xgb_0/xgb_1`，先过完整树重放门槛。见[d33-shallow-replay-plan.md](d33-shallow-replay-plan.md)。

## D34：E1质量变体

`brier_crisis`：`Y_g=n_g/N`，`A_g=(n_g−Σℓ_g)/N`，`ℓ=(p_crisis−z)²`，概率取当前父模型；`get_c_b`/`scan`不改。见[d34-e1-brier-contrast-plan.md](d34-e1-brier-contrast-plan.md)。
