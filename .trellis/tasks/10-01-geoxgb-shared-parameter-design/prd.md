# GeoXGBoost 部分参数共享：需求规格 v1.0

状态：**当前：D53/A27零拟合精确起点状态概率水平转移诊断已批准（d53-state-probability-transfer-plan.md）；D52/A26独立二分类目标root诊断已完成、不采用（核验通过；用户批准的仅诊断，不改四分类主模型；d52-binary-root-proposal.md）；D51/A25历史+日历（78特征）root消融已完成、不采用（核验通过；d51-history-calendar-root-plan.md）；D50/A24零拟合精确起点阶段排序诊断已完成、不采用任何策略（核验通过；d50-origin-phase-ranking-plan.md）；D49/A23零拟合排序余量诊断已完成、不采用任何策略（核验通过；d49-ranking-headroom-plan.md）；D48/A22已知起点FIT限制对照已完成、不采用（核验通过；d48-known-origin-fit-plan.md）；D47/A21深度1（stump）root对照已完成、不采用（核验通过；d47-stump-root-plan.md）；D46/A20 2×危机类加权root训练对照已完成、不采用（核验通过；d46-crisis-weight-root-plan.md）；D45/A19保存root前缀学习曲线诊断已完成、未采用前缀/早停策略（核验通过；d45-root-prefix-diagnostic-plan.md）；D44/A18零拟合full-pool vs r80-FIT root诊断已完成、不采用full-pool替换（核验通过；d44-full-pool-root-diagnostic-plan.md）；D43/A17时间块Brier地图学习、共同重拟合对照已完成、不采用（核验通过；时间切分/地图生成变体序列按预设停止；Stage1仍未解决；d43-temporal-map-refit-plan.md）；D42/A16旧/当前地图共同重拟合迁移对照已完成，两臂均不采用（175次区域L1，核验通过；d42-map-transfer-plan.md）；D41/A15冻结分区局部增量减半对照已完成，诊断候选、不采用（零拟合，核验通过；d41-local-shrinkage-plan.md）；D40/A14起点前决策规则可行性对照已完成、不采用（无新拟合，核验通过；d40-forward-decision-plan.md）；D39/A13概率诊断已完成（仅诊断，核验通过；d39-probability-diagnostic-plan.md）；D38/A12固定弱persistence margin root已完成，核验通过，保留为探索性root候选、不采用为默认（2d4fe4e；d38-persistence-margin-root-plan.md）；D37/A11时间加权root已完成且不采用（3b53989，核验通过；d37-recency-root-plan.md）；D36已完成（仅分析）；D35/A10已完成（be5f485，核验通过；Brier仅为探索性候选，D35后停止新训练；d35-global-increment-control-plan.md）；D34/A9已完成（7b2bf6f，核验通过）；D33/A8已完成（69f2cc3）；D32/A7分配证据导出已完成（c079b75）。D30/D31已完成（D31不确定，见各自计划）。Stage1过拟合仍未解决；审计run ed632775保持active，本轮不运行Stage2/3、完整648、最终评价或关闭。**
brainstorm → spec → grill 已完成；D24确认设计，随后“可以开始执行”授权按冻结规划实施及首轮有限实验，取代此前仅规划的范围。此处状态文字不代替task.json或审计运行记录。
基准 commit：`14c89bc150194452361bb495c601de070cd94ce7`。

## 目标与价值

用可复现的有限探索，检验冻结共享 XGBoost 树及受限局部增量能否改善食物危机预测、减少分区过拟合。既要追求超过 persistence/expert，也要能把无效分区、数据支持不足和负结果准确地报告出来。

## 需求

- R1：从固定commit的 `FEWSNETFourClassBaseline/` 四分类包建立隔离的 `FEWSNETGeoXGBExperiment/`，保留原基线、输入及产物。母包、55个源／包文件的边界与独立代码/运行身份见implement.md（D1/D24/D25）。
- R2：底座换成 XGBoost；partition 学习时冻结并继承当前父模型的已学得树，只追加受约束的局部树；最终预测各分区共享同 fold 的全局树并学习局部增量，允许零增量回退。相同超参数不等于共享学得参数。容量、支持资格、启用规则与验证日程统一以 experiment-plan.md 为准（D4/D13/D24）。
- R3：按 D26，当前主评价终点为四分类 argmax 折叠后的 **crisis-positive F1**，fixed-four macro-F1 为次要结果。最终仍要求 H4/H8/H12 各自超过同口径 persistence，配对增益95% CI下界均 >0；不能用平均成绩抵消任一 H 未达标。超过 expert 是 H4/H8 优化目标，H12 不造 proxy。最终下游契约对齐及执行仍按 experiment-plan.md A1 等待审阅，D27 不授权进入最终评价。
- R4：尽可能减少过拟合，重点控制分区搜索及额外局部模型复杂度。按 D15，每个 H 在开发期统一选择全局配置与局部增量配置各一套，选定后所有分区共用，不逐区或逐节点调参；各区仍拟合自己的增量树，并按 D12/D13 选择启用或零增量回退。
- R5：规划已收敛；D25授权按implement.md实施、验证，并按experiment-plan.md执行首轮有限开发与冻结后的最终回溯评价。先提交规划、绑定实际Claude执行会话并通过审计wrapper启动，保持数据/模型/预算边界。
- R6：保留三阶段主线：共享参数的 XGBoost 重新学习候选 partition，沿用既有 consensus 方法生成新地图，再按新地图进行全局共享树＋局部增量的滚动预测。旧 RF 地图保留为固定地图对照（D5）。
- R7：expert 仅作比较基线；其预测及派生值不作为模型特征、预测纠偏或融合输入。预测起点可用的真实 IPC 及其历史仍可保留，不能与 expert projection 混同（D6）。
- R8：固定母包既有 162 列特征的来源、定义、顺序和 origin 对齐，保留真实 IPC 历史及现有缺失指示列，不新增／删除字段或改变工程公式。新 XGB 路径按 D14 保留 NaN、使用树模型原生缺失路由，取消模型侧 RF max_plus 填补；保留 ±inf→NaN 清洗和真实零值。全局、父、子及预测路径采用一致输入表示（D7/D14）。
- R9：区分区域误差/q 搜索、当前父模型的分割接纳、Stage 1 月度评分/consensus、开发选择及最终评价。Stage 1 E1/q 与 E2/接纳复用区域内随机验证行，仅作候选生成筛选；E2 按 D23 比较严格 gain >0 和 >0.01 两族。Stage 3 仍在该区同键历史时间验证上相对 global 严格增益 >0.01 才启用局部增量。两层总体及阈值族分开，均非显著性检验；外部评分信息边界保持（D8/D10/D12/D13/D20/D23）。
- R10：实际标签窗口固定 W=59，定义为 `[O-59,O)`，适用于新设计各 H 的全局／父／局部拟合；内部时间验证在各自 origin 完整回溯，不机械截短为当前外部 fold 窗口的子集。取消此前 35/59 两档择窗，不按区、H 或最终分数选窗口；不改变 162 列特征定义及其自身历史公式（D9 修订）。
- R11：接受局部训练／验证缺少“4或5”是现有数据现实，不以该类缺失本身自动禁用局部增量，也不要求每区四类齐全。保留固定四类输出和主指标口径，记录各类支持及无法验证的类别表现；其余支持资格仍适用（D11/D24）。
- R12：2018–2020 目标月用于开发及地图学习，最终配置与地图选择仅用截至 2020-12 已可用信息。之后冻结超参数、地图和自动路由规则，按母包各 H 原起止日程滚动评价至 2024-12。后续已到达标签可按固定 59 个月规则更新拟合及 D13 路由，不得据最终成绩重调参数、阈值或地图；报告为回溯评价（D16）。
- R13：允许用当前外部 origin 已知的地图组织更早历史验证，作为局部启用筛选，承认地图选择偏差；不要求地图在每个内部 origin 已存在。最终地图只用截至 2020-12 的开发信息生成，Stage 3 的标签／成绩不得参与地图、超参数或阈值选择；D16 已批准的到达标签滚动拟合／固定规则筛选继续适用，本次尚未到达的目标标签不得使用（D18）。
- R14：Stage 1 的核心是为 Stage 2 生成多份候选分区，采用区域内随机80/20与50/50；Stage 3 继续 rolling-origin。候选生成、质量评分／共识权重和最终评价分开。按 D24 固定 seeds 与开发预算，单独核查候选数量、重复及权重集中，不把候选数当独立样本量（D19/D20/D24）。
- R15：候选质量、结构稳定性与多样性服务于最终预测表现，不单独最大化候选数／差异。高度一致可作为稳定结构的信号，但不自动证明正确；不为强行制造差异临时放宽门槛或排除一致候选。D23 的两档门槛属于预先声明的有限对照。方案取舍由开发期模拟 Stage 3 的完整流程成绩评估，最终 Stage 3 不参与选择（D21）。
- R16：本任务基调是探索性实验；允许提出有科学问题的有限开发对照，记录正／负结果及下一步依据。区分实现正确与科学目标达成；未超过 baseline 时如实报告，不能伪称达成。已确认的信息边界、特征契约及科学成功标准仍有效，拟修改的设计须显式列为待审；探索性不代表可以扩充本轮预算或改变冻结契约（D22/D25）。

“必须超过”不是结果保证；实验失败应如实报告，不得改测试集、比较键或指标来制造成功。

## 背景与证据

| 候选母包 | 任务口径 | 对本任务的意义 |
|---|---|---|
| GeoRFBaseline | 二分类 GeoRF source release，F1 objective、无 SMOTE | 干净母包，但旧 record shift 不等于正确月历对齐 |
| FEWSNETCleanPersistenceExperiment | 二分类 persistence 的保守 0→1 修正 | 最终 ABDE 地图未分区，partitioned 实际复用 pooled |
| FEWSNETFourClassBaseline | 1/2/3/4或5；GeoRF、pooled RF、persistence、expert | 四臂口径完整；13-cluster 局部 RF 比 pooled 差 |

出处：`GeoRFBaseline/README.md:129-138`、`GeoRFBaseline/src/preprocess/preprocess.py:267-272`、
`FEWSNETCleanPersistenceExperiment/runs/20260920_stage1/report/final_report.json:251-290`、
`FEWSNETFourClassBaseline/README.md:3-37`。用户已选择四分类母包；代码 fork 尚未创建，D25已授权执行，待审计wrapper启动。

已提交四分类 v7 的 fixed-four macro-F1，仅读取现有 CSV，未重新评分：

| H | partitioned RF | pooled RF | persistence | expert |
|---|---:|---:|---:|---:|
| 4 | 0.6532 | 0.6834 | 0.7308 | 0.7651 |
| 8 | 0.5894 | 0.6102 | 0.6662 | 0.6980 |
| 12 | 0.5337 | 0.5522 | 0.6406 | 无对应 expert |

来源：`FEWSNETFourClassBaseline/runs/fourclass-v7-20260928/report/arm_metrics.csv:2-18`。
局部模型较差不单独证明经典训练集过拟合。Stage 1 反复在 area 内随机 validation 搜索分区，存在选择过拟合风险
（`FEWSNETFourClassBaseline/README.md:107-117`）。

本轮已沿实际四分类代码区分评价层：q 搜索与子模型接纳目前共用 `X_set==1` 行；后者是相对当前父模型的 macro-F1 严格增益 >0.01，并非 p-value 检验；只有首层父模型为全局 root。
Stage 1 外部目标月分数另用于 consensus 权重，Stage 3 最终比较及置信区间又是独立用途。完整公式、调用点和边界见 `evaluation-contract.md`。

按用户要求已核查真实样本支持，见 `research/sample-support.md` 与计数脚本／摘要。Stage 1 每行政区历史标签中位数 8、当前随机验证标签中位数 2；Stage 3 每分区 fitting 行数 1,016–7,728，但只有 8 个真实标签月份，390 个分区×fold 中 156 个没有“4或5”类。当前 ≥50 行／≥2 类门槛未让任何这类 local 回退，不能据此认定支持充分。
在旧窗口内拆出两个6个月验证块只得到2个和1个标签月份，因此撤回该硬切方案。更早 rolling-origin 历史存在；W=59可增加真实标签日期，但不能保证性能。早期特征覆盖已做只读计数，部分行149/162列缺失，保留原NaN并披露；数值兼容及预测性能留待实施后验证，不能把计数当成训练证据。

旧 GeoXGB 逐分支新建 XGBClassifier，没有共享学得树（`src/model/model_XGB.py:133-140,234-268`）；
旧父模型回退限定 RF/DT（`src/partition/transformation.py:733-757`）。不能把旧 adapter 当作满足 R2/R4 的现成实现。

当前四分类的 162 列特征不含 expert projection，已包含 exact-origin 的真实 IPC phase 及其历史。
用户已按 D6 确认 expert 仅作对照，并按 D7 确认固定现有 162 列特征定义与时间对齐
（`.trellis/tasks/archive/2026-09/09-28-fewsnet-four-class-perturbation/feature-contract.md:17-21,47-52`）。

证据边界：v9 未验证、未提交，已提交证据仍 v7；当前代码不等于 v7 的同次运行，也不继承审计通过结论
（`.trellis/tasks/archive/2026-09/09-28-fourclass-audit-repair-4/IMPLEMENTATION_LOG.md:44-51`）。

## 验收标准

- A1 / R1：明确母包、source commit/release、fork 边界及复现基线；原包/输入/产物不覆盖。
- A2 / R2：按 D4 验证父/全局共享树的结构和叶值保持不变，局部只追加受约束的增量树。Stage 3 保存该区历史时间验证的 global/global+local 同键预测、得分与差值，严格增益 >0.01 才启用；等于 0.01、更低或没有可用验证收益证据时为零增量，复现全局预测。模型记录共享来源、新增容量、实际路由及回退原因，预测可重放；不使用本次目标标签选路由（D13）。
- A3 / R3：对每个 H∈{4,8,12}，在该 H 相同且预先固定的评价键上，Δ_H = macro-F1(主候选 GeoXGBoost) − macro-F1(persistence) 必须严格 >0，且 D17 国家块配对 bootstrap 的95%区间下界严格 >0。使用共同国家抽样权重，先合并 confusion counts 后重算 F1 和差值，保留固定四类轴；seed=42，2,000次有效抽样，最多20,000次尝试，linear 2.5%/97.5%分位数。任一 H 未达标即整体未达科学目标；区间不足为 incomplete。无额外 +0.01 最终增益要求，不把三个边际区间称同时95%；区间解释限于固定预测与已观察历史，Expert 仅比较合法 H（D2/D3/D17）。
- A4 / R4：按 D24 固定预算比较 pooled、同旧图独立/共享 local、新图共享 local；保存训练/开发逐键预测、支持与容量，报告训练/开发差距及跨时间稳定性。每 H 最终全局／局部配置各只有一个身份，跨分区、分割层及最终 folds 复用，不逐区或逐节点调参（D15）。按 D12/D23、D13 分别接纳，不把逐区通过当成整体 macro-F1 必然提升。独立臂只作固定配置诊断，历史35个月RF不作匹配窗口的backend因果对照。
- A5 / R5：grill及最终规划复核已完成，prd/design/experiment-plan/implement与真实上下文清单完整一致；D25后按批准顺序实施/验证/实验。记录实际start、执行会话、base SHA及可重放证据；不把D24设计采用单独当作执行批准。
- A6 / R6：主候选使用共享 XGBoost 的候选分区证据，经沿用的 consensus 方法生成新地图；保存来源、时间边界和地图身份。旧 RF 固定地图对照独立标识，不与主候选结果混用；新图分区数由沿用的 consensus 规则决定（D5）。
- A7 / R7：特征来源和预测链中无 expert projection 或其派生输入；expert 仅进入相同评价键上的比较结果。真实 IPC 历史按自身 origin 可用性检查，不因 D6 被排除（D6）。
- A8 / R8：162 列有序 schema、源字段、工程公式和 origin 规则与冻结母包一致；新 XGB 路径仅统一将 ±inf 转 NaN，保留已有 NaN、真实零与其他有限值，dense 输入设置 missing=NaN，不拟合 MaxPlusImputer、不增删指示列。共享前缀的缺失值分支方向也必须保持不变；训练、增量和重放使用同一输入约定。原 RF 包及其结果不改写（D7/D14）。
- A9 / R9：逐层留存评价行键/时间角色、训练可用标签、模型/父 checkpoint 身份、q 与扫描得分、接纳得分、consensus 权重和最终结果。Stage 1 fitting／随机 validation 键互斥且来自候选外部 origin 前合法池；E1/E2 复用，不称逐内部起点前瞻重放。E2 在完整父验证键上汇总 confusion，按候选身份的严格 >0 或 >0.01 接纳，恰等于门槛均拒绝，父模型赢平局。Stage 3 另守各内部 origin 与 >0.01 门槛；不把 E2 规则套到 q 或最终 CI（D8/D10/D12/D20/D23）。
- A10 / R10：每个新设计拟合身份记录 W=59、对应 `[origin-59,origin)`、真实 fitting keys 及可用特征支持，不保留逐区或逐 H 择窗逻辑。旧 35/59/71 计数仅为研究证据；既有 35 个月 RF 结果不冒充与新 59 个月模型严格同窗的底座对照（D9 修订）。
- A11 / R11：缺“4或5”类的分区仍可按其余已冻结资格和接纳规则进入增量候选；输出类别轴及 fixed-four macro-F1 不随局部观察类别改变。记录真实各类支持，缺真例时不声称该类召回率已被验证；不因缺类本身触发强制回退（D11）。
- A12 / R12：记录最终配置／地图的2020-12信息截止；最终排期分别为H4=2021-05、H8=2021-09、H12=2022-01起至2024-12，空目标月不造标签。最终每折拟合/自动路由只用其origin可用信息，参数/阈值/地图冻结。开发构图按origin截断候选评分目标；D24 已接受先全开发期选G的顺序筛选，开发分数须标明配置选择偏差，不声称配置也在每个历史origin已知。披露已查看历史基线，结果称回溯评价（D16/D24）。
- A13 / R13：地图来源记录和选择日志证明最终地图／配置只用截至 2020-12 信息；每条内部预测记录自身 fitting origin，每次路由记录当前外部 origin 与已到达的验证目标月。条件历史分数仅用于内部筛选，不称独立前瞻效果；最终目标标签只进入本次预测锁定后的评分。后续 fold 依法使用已到达的早期标签，不据 Stage 3 报表修改地图或选择规则（D18，与 D16 一致）。
- A14 / R14：Stage 1 候选记录比例、H、目标月、G/L、seed、门槛族与样本键身份；80/20和50/50均保留，E1/E2用于内部生成／接纳，E3经E4进入共识。按D24核对648候选任务、24方案及完整预期账本；报告总数、正权重、同覆盖分区重复与权重集中，不强行去重或凑多样性，最终Stage 3不参与择优（D19/D20/D24）。
- A15 / R15：不把最低不同分区数、低相似度或高 seed 数设为科学成功条件；正权重与一致性诊断和开发预测分数分开报告。Stage 2 沿用共归属汇总／聚类，可生成不同于任一候选的新图，不宣称算法保证找出预测最优分区。方案选择只读取开发证据，最终信息截止和科学验收独立遵守 D16/D18（D21）。
- A16 / R16：每组新增开发对照注明研究假设、固定变量、比较指标与最大预算；先用开发期比较，最终候选／规则冻结后再进入最终评价。实现／复现检查与 baseline 科学胜出分别给出状态；开发期可有界迭代，查看最终结果后的改动归入新的探索，不沿用原最终评价的独立性表述（D22）。

## 当前范围之外

不扩充批准的搜索网格或依据最终成绩重调；不修复旧审计、发布新模型；不引入Ethiopia-only、IPCCH、same-row expert、fs3 expert proxy、旧binary flip。
Expert 输入、基于 expert 的预测纠偏和融合已按 D6 排除；新增/删除特征或改变工程公式已按 D7 排除。Ordinal loss 和更换 consensus 算法均不是已批准设计。

## 已确认决策与修订记录

以下保留用户逐次决定的原意；其中“尚未确定”等描述指当时状态。后续 D20 取代 Stage 1 内部时间方案、D23 扩展 Stage 1 接纳族、D24 冻结首轮数值及预算；当前执行契约以以上需求/验收和 experiment-plan.md 为准。

- **D1 — 2026-10-01**：用户对“以 FEWSNETFourClassBaseline 四分类包作为 fork 母本”回答“可以”。选择四分类母包，排除二分类 clean-persistence / 原二分类 GeoRFBaseline 作为本次直接母包；不代表共享机制、验收门槛或实施已获批准。
- **D2 — 2026-10-01**：用户确认“4、8、12 个月三个预测期的 macro-F1 都必须严格超过同口径 persistence”。确认逐 H 胜出，不是三者总体平均胜出；增益强度另见 D3。
- **D3 — 2026-10-01**：用户对“各预测期相对 persistence 的 macro-F1 配对增益，其 95% 置信区间下界均 >0，暂不另设 +0.01 最低增益”回答“采用”。该门槛是硬验收要求；重采样单位及区间适用范围后续已由 D17 冻结。
- **D4 — 2026-10-01**：用户采用“冻结已有树，局部只追加少量受约束的树”：partition 时子分区继承父树；最终预测时各分区共享全局树并学习局部增量；允许零增量回退。接受以限制新增容量为代价约束局部拟合自由度。具体新增轮数、深度、正则和回退判定尚未冻结。
- **D5 — 2026-10-01**：用户采用三阶段主线：共享参数 XGBoost 重学 partition → 沿用既有 consensus 方法生成新地图 → 按新图进行共享全局树＋局部增量的滚动预测；旧 RF 地图保留为对照，用于区分模型变化与地图重学的贡献。重学地图属于主线，不再是可选的后续扩展；运行及调参预算尚待固定。

- **D6 — 2026-10-01**：用户采用“expert 仅作比较基线，不进入模型特征、纠偏或融合”的信息边界；允许保留现有真实 IPC 历史。该决定用于评价 GeoXGBoost 独立于 expert projection 的预测能力，不授权额外 expert-assisted 实验。
- **D7 — 2026-10-01**：用户回答“固定”，确认本轮固定现有 162 列特征的定义、顺序及时间对齐，保留真实 IPC 历史，将优化集中于 XGBoost、共享机制及 partition；缺失值处理另行确定。
- **D8 — 2026-10-01**：用户采用时间验证，同时明确提醒区分“区域内部评价以计算 q/决定分割”和“分区模型相对根模型的显著性评价”等不同层。按当前母包事实补充：后一层实际比较当前父模型，采用 >0.01 性能门槛；独立根模型比较及最终区间各有用途。用户此答复未自动批准将 q 搜索集与接纳集拆成两套，也未批准在内部增加统计显著性检验。
- **D9（修订）— 2026-10-01**：用户先接受 35/59 两档，随后明确“修正之前的选择，直接用59个月”。当前契约为固定实际标签窗 59 个月，取代两档开发比较；不改变既有 162 列特征公式，不授权训练。35/71 个月仅保留为已完成的支持度计数参照。
- **D10 — 2026-10-01**：用户对 E1/q 与 E2/接纳复用多起点时间验证数据、接纳仅作内部筛选、由外层时间评价检验整体流程的取舍回答“接受”。当时未恢复区域内随机验证；**后由 D20 明确取代 Stage 1 多起点安排**，复用及选择偏差原则保留，不代表内部接纳获得独立显著性。
- **D11 — 2026-10-01**：用户说明“缺少4或5目前只能这样，因为4或5本身就是稀少类别”。接受局部缺该类，不将其本身设为增量模型硬禁用条件；保留真实支持记录、固定四类口径和未验证类别表现的边界，不据此放开其他过拟合约束。
- **D12 — 2026-10-01**：用户对“Stage 1 相对当前父模型，macro-F1 严格提升 >0.01 才接纳，否则保留父模型”的建议回答“采用”。固定为内部性能筛选，接受可能舍弃小幅真实增益的代价；不是 q 门槛、独立显著性检验或 D3 最终 CI，也未自动批准 Stage 3 使用同一规则。**D23 后扩展为 >0 和 >0.01 两个固定生成族**，不再只有一个 Stage 1 门槛。
- **D13 — 2026-10-01**：用户对“Stage 3 在每区同一组历史时间验证样本上比较全局＋局部增量与全局模型，macro-F1 严格提升 >0.01 才启用，否则零增量回退”的建议回答“采用”。这是独立于 D12 的路由选择规则；等于 0.01 不启用，不能使用本次待预测目标标签，不因缺少‘4或5’本身禁用，整体仍按 D3 验收。
- **D14 — 2026-10-01**：用户对“保留 NaN，由 XGBoost 原生学习缺失分支，取消模型侧 RF 填补，保留既有特征公式、真实零及缺失指示列”的建议回答“采用”。保留母包已有 ±inf→NaN 清洗；不在子区另拟合填充值，不借此改动原 RF 包或特征语义。共享树的既有缺失路由随前缀冻结，性能影响仍须开发评价，不授权本轮训练。
- **D15 — 2026-10-01**：用户对“每个预测期统一选择一套全局超参数和一套局部增量超参数，所有分区共用，不逐区调参”的建议回答“采用”。全局与局部配置可不同，选定后跨分区／分割层及最终 folds 复用；各区仍学习自己的增量树，并按 D12/D13 决定启用或回退。这限制的是超参数搜索自由度，不替代 D4 共享已学得树，也不固定具体容量数值。

- **D16 — 2026-10-01**：用户对“2018–2020 开发和地图学习，随后冻结参数与地图，沿母包日程滚动评价至 2024 年底；已到达标签仍按 59 个月规则参与拟合及局部启用，不根据最终成绩重新调参”的建议回答“采用”。固定阶段边界与回溯评价表述，最终配置／地图选择信息截止为 2020-12；具体内部和开发外层 origin 日程仍须设计，不代表本轮可启动实验。

- **D17 — 2026-10-01**：用户对“最终置信区间沿用国家块配对 bootstrap，整国全部地区和月份一起抽样，GeoXGBoost 与 persistence 共用抽样权重；保留国家内部时空依赖，不涵盖未知未来年份冲击”的建议回答“采用”。保留 D3 各 H 的95% CI下界 >0 要求；不扩展成每区显著性检验，不把历史期条件区间表述为完整未来不确定性。

- **D18 — 2026-10-01**：用户在具体 H12 例子解释后确认“这个可以的。只要stage3的那些信息确保没有用到就可以，地图本身是可以用之前的信息生成的”。采用当前已知地图下的条件历史筛选，接受其内部选择偏差；最终地图只用截至 2020-12 的开发信息，Stage 3 评价信息不参与建图／调参。与已确认 D16 衔接：后续起点已到达的历史标签仍可按冻结规则用于拟合和 D13 筛选，未到达目标标签禁止使用；不新增冻结所有模型于 2020 年的限制。本次仍仅批准设计取舍，不批准实施／训练。

- **D19 — 2026-10-01**：用户强调“一阶段的核心是通过生成多份候选来供二阶段进行共识训练”，训练方式可以不同，“比如拿一半训练一半验证也是可以的”。确认 Stage 1 候选多样性与 Stage 3 预测验证须分开设计；50/50 是用户认可的可能方式，尚不指定随机、时间块或按区域划分。Stage 1 内部时间验证的先前方案因此重新进入设计，不自动以该表述批准某个随机算法；E3/E4 的评分用途及 Stage 3 无未来信息边界不变。

- **D20 — 2026-10-01**：用户对“Stage 1 在合法 59 个月历史池内，区域内随机 80/20 与 50/50 两类划分生成候选；Stage 3 继续 rolling-origin”回答“可以”。此项取代 D8/D10 先前对 Stage 1 内部多起点时间验证的安排，保留 E1/E2 复用、D12 接纳、E3/E4 外部评分及 Stage 3 时间边界。随后用户强调候选数量与多样性可能不足，故 seed 数／总候选预算尚未确定；同意两类划分不等于批准只用一个 seed。

- **D21 — 2026-10-01**：用户指出高 confidence 的 Stage 1 分区可能接近或相同，需要在质量、一致性和多样性之间取舍，最终目标是 Stage 3 表现以及超过 persistence 和 expert。据此不将候选差异／数量作为独立优化目标，一致候选不自动视为失败；超过 expert 明确为优化目标，是否套用 D3 同样的硬 CI 标准待确认。本答复未单独批准五个 seeds 或具体候选总预算。

- **D22 — 2026-10-01**：用户表示“你有什么想法也可以提出来，这个task的基调是探索性实验”。据此按研究假设与有限对照组织后续规划，避免把未经实验支持的容量／候选数量当理论充分条件。已确认约束仍生效；本答复不是对任何新增参数、阈值或训练执行的批准。

- **D23 — 2026-10-01**：用户对新增“Stage 1 严格 gain >0 与 >0.01 两档接纳门槛的对照”回答“可以”。扩展 D12 为两个固定候选生成族，父模型赢平局，E3/E4 仍按原规则评分；Stage 3 的 D13 保持严格 >0.01。允许检验宽松生成的收益／噪声，不授权其他阈值或无界搜索，也未批准具体 seed 数、容量和总预算。

- **D24 — 2026-10-01**：用户对“采用整套有限方案作为第一轮探索设计，包括全开发期先选全局配置造成的选择偏差；此次只确认设计”回答“采用”。确认 experiment-plan.md 的4套G、2套L、3个split seeds、648候选任务/24完整流程方案、80-round路径上限、真实支持底线、六日期历史gate、配置选择顺序、比较臂及停止规则。接受开发选择偏差，最终2020-12信息截止不变；expert保持H4/H8优化目标及差值/区间报告，不新增硬CI要求。此前该文件中的数值/预算待审事项至此解决；未授权代码fork、实施、训练或审计启动。

- **D25 — 2026-10-01**：最终规划已呈现后，用户明确“可以开始执行”。授权按v1.0冻结规划实施、必要契约验证及首轮有限实验，包括隔离fork、开发择优和一次冻结后的最终回溯评价。此前仅规划/未授权执行的状态至此被取代，科学契约、预算和截止不变。遵守AGENTS.md：批准规划先提交，实际Claude执行者在Herdr绑定并通过审计wrapper启动；不授权扩充实验或绕过生命周期。

- **D26 — 2026-10-01（范围修订，执行中由用户明确授权）**：用户在看到开发期诊断（pooled XGB 在1–3类不逊于persistence，但几乎不预测稀有“4或5”，宏F1差距几乎全部来自该类）后指示：“可以的，先解决一阶段问题。可以保留四分类概率但按照二分类评估。”据此：(1) 优先解决 Stage 1，停止旧四分类目标的批量执行（已停止，保留全部产物为被取代证据）；(2) 保留原生四分类 XGB 概率（multi:softprob、num_class=4、固定类轴），评价终点改为二分类危机 crisis = IPC≥3（类3与“4或5”）；预测规则为当前四分类 argmax 后折叠（实测优于概率和≥0.5）；(3) 主诊断/目标为 crisis-positive F1，四分类 fixed-four macro-F1 保留为次要；(4) Stage 1 的 E1 扫描、E2 完整父验证键精确增益、E3 目标月得分及 E4 权重输入统一改为该二分类终点；(5) 用已有72个 G 预测在开发期同键上按二分类 F1 重选冻结的4套 G 之一，不新增 G 拟合；(6) 59个月窗、支持底线、续训前缀/80轮上限、两门槛族、预算与不使用最终数据均不变，除非诊断出具体 Stage 1 问题并另行记录修订；(7) 先在正确几何的有界代表性子集上修复/测试 Stage 1，报告 E2-vs-E3 泛化与逐类/二分类混淆；在用户审阅 Stage 1 结果前不启动二分类 Stage 2/3。下游需对齐的指标（地图权重、Stage 3 局部 gate、开发择优、D3 最终区间）写入 experiment-plan.md 修订节 A1，待审后才执行。D2/D3 的“persistence 同口径”比较在该修订下理解为同一二分类终点；D24/D25 记录保持原样作为历史。

- **D27 — 2026-10-01**：用户对“沿用两个目标月、三个 H、L1 和 gain>0，仅改成最近三个真实标签月份的时间验证，共六个根任务”的建议回答“可以的。记得修改对应的spec，prd，implement等文档”，随后强调“但我们stage1过拟合的问题还没解决吧？”。批准同步规划与执行该有界对照，不是认定问题已解决。保留 D26 随机结果作为对照；新实验 E1/E2 仍共用一个时间块，不是独立确认集或严格内部 rolling-origin。完整数值、日期、配对和停止规则见 experiment-plan.md A2。要求分别报告实现检查、实验完成、泛化改善证据；完成后再次审阅，不自动启动完整648、Stage 2/3 或关闭整个任务。

## 收敛与交付状态

**D27 执行状态（2026-10-01）：** 实现完成（3dab25b，native check 无影响结果问题，52 测试通过）；六根运行完成；科学证据：tb3 根因移出最近三个月显著变差（root crisis F1 平均 −0.056），local 增量的迁移在 5/6 对略好但 1 对大幅失败，未显示可靠改善——Stage 1 过拟合仍未解决，待审阅。

**当前验收补充 A17 / D27：** 六个预定时间块候选必须有完整身份、实际 fitting/validation 键和同键 E3 root/local 预测，并与既有六个 r80/L1/gt0 对照比较；记录支持与结构多样性，报告根模型本身变化。E2 上升、训练完成、Trellis check 通过均不能作为“Stage 1 过拟合已解决”的证据。D27 至多提供有限开发对照证据；最终 persistence/expert 目标未完成。

设计范围、科学验收及首轮预算已收敛，最终规划已呈现并获D25执行授权。design.md、evaluation-contract.md、experiment-plan.md、implement.md和真实上下文清单构成冻结规划包。

执行前先验证XGB3.0.0固定四类续训/前缀不变与重放、资源、运行环境和证据链；不通过则停止依赖工作，不自行换模型/日期/预算。实际状态由task.json及审计记录确定；D25授权执行，D24单独仍不构成执行批准。

## Plan B / D28：根模型约束的局部修正

用户要求：“可以brainstorm考虑plan B，先解决这个问题再说。” 本阶段重新讨论 Stage 1 局部修正的泛化机制，保留候选多样性及最终 Stage 3 目标；不要求每个探索候选都胜过root，也不把零分裂或E2高分当问题已解决。

已确认事实：D27没有显示可靠的E3改善。H4/2018-02的同键E3 root→local：TP433→421，FP136→303，FN339→351；E2为TP1091→1181、FP706→883。终端0100/01011/11101合计解释净增加167个FP，决策记录推算累计增量80/80/60 rounds（未额外重放模型树数）；其中11101的危机占比从E2的53.66%降到E3的10.31%。同area的E2/E3路由一致。证据为外部运行 `geoxgb-d27-tb3-20261001/stage1_tb3/candidates/h4_2018-02_G1_L1_tb3_s42_gt0/{candidate.json,target_predictions.csv,validation_predictions.csv.gz}`。这支持检验累计修正与时间不稳定性，不证明具体因果机制。

方向取舍：

1. **根模型约束的局部修正（D28已采用方向）**：允许继续细分空间，但每个区域始终从同一root出发，只拟合一次小增量，不继承并累计所有祖先增量。保留全局已学得参数共享；仅对该新Stage1臂取代D4的逐层父增量继承，旧对照不改。先固定既有数据划分/局部配置比较，不能同时将增量轮数、学习率、门槛和验证方式全部改动。
2. **压缩局部修正幅度**：保留原递归架构，减少累计容量或采用全局统一的增量收缩；不逐区挑缩放权重，不用E3或Stage3为各区域调参。此方向更接近现有机制，但可能只把有害和有益修正一起压向零。
3. **隔离分区搜索与接纳证据**：另设诚实确认或交叉拟合，针对E1/E2选择偏差；真实月份有限，须评估支持、样本时效和代价。不是把D27现有时间块换名就获得独立确认。

**D28（方向及执行采用）**：用户先对方向一回答“可以按照一”，随后对最终呈现的六根实施/实验方案回答“接受”。已授权按A3实施、检查并运行六根r80/L1/gt0对照；共享root、一次局部增量，保留原搜索预算上限，代价是可能放弃真实层级修正收益。方向二/三暂不实施，不因机制采用声称过拟合已解决，也不扩展到Stage2/3或完整648。

**D28 执行状态（2026-10-01）：** 实现完成（98adf48，native check 无影响结果问题，58 测试通过）；六根运行完成且与 D26 对照 root 完全相同；科学证据：共享root使 local−root 均值 −0.0004（父链 −0.0014），4/6 对更好但 5/6 仍 ≤0，E2 乐观缩小——未产生正向外部增益，Stage 1 过拟合仍未解决，待审阅。

**A18 / D28 验收补充：** 新子模型只含共享root及一次L1增量；E2仍比较当前父路由。未接纳侧或支持不足侧保留当前父预测，不强制回root。共享来源、比较对象、实际新增容量和沿路径搜索额度分开记录。六组同键E3及候选多样性作为有限开发证据，不要求每个候选都胜出，不用E2/测试通过替代泛化证据；最终persistence/expert目标和Stage3截止不变。

## D29：冻结候选后的独立确认（已完成，待科学评审）

用户先回答“可以，就按照推荐讨论”，随后对“首轮独立确认只做诊断”的范围回答“采用”，并进一步指示“没问题stage3以外的数据怎么搞都行”。据此开发期数据拆分/复用由执行方安排，无需反复逐项确认；本轮具体采用A4六根预算推进实施和运行。确认保留原fitting/root、将原validation分为搜索S及确认C、先冻结完整候选再在C评价；C不删候选、不剪枝、不触发回退。Stage3数据继续隔离，不用于本轮选择。只读调查见 `research/e1-e2-confirmation-feasibility.md`。

已确认：E2的接纳结果改变后续E1的区域、父模型及继续搜索，因此仅把E1/E2样本拆开仍存在后续自适应复用。现有r80 fitting行可保持不变，将原validation按预定无标签规则分成搜索/确认两半；按旧地图预检，80个有validation的终端中79个两半满足原支持底线，但确认侧26个终端危机正例不足20（仅诊断，不新增门槛）。旧地图是用完整validation搜索得到的，该预检不构成诚实确认，必须重新搜索并隔离确认标签。

**已采用范围：** 搜索侧保留原E1/q和E2接纳，先冻结整份候选地图、模型和路由，再一次性读取确认侧，随后看原目标月E3。首轮确认只诊断、不据此删图/剪枝或回退，不改变E4。S/C/E3差异用于定位问题，不作因果分解或显著性证明；C只相对本次新搜索保持隔离，既有开发数据/G选择已被使用，不称全研究从未触碰的测试集。

具体方案见experiment-plan.md A4、design.md D29、evaluation-contract.md D29及implement.md §9：新增原六H/T组合的六根/六候选、root单次L1/gt0，D28为主对照。样本不够时沿用原搜索支持回退，不补C、不放宽门槛。先提交规划再由原Claude执行；批准不等于已完成，完整任务仍未完成。

## D30：近期搜索、完整历史拟合

用户授权“可以，按照你推荐的额方式研究”，并已授予Stage3以外开发数据安排。执行d30-recent-search-plan.md六根方案：保留D29原fitting/root/C及模型设置，仅原S最近六个真实标签月用于E1/E2，原S更早行不用、不并入其他角色。C保持只诊断，主对照D29，不放宽支持。目标是检验时间较近的误差是否更有用；不预设改善，不接触Stage3。

## D31：逐区等量搜索对照（已完成，待研究综合）

监督方决定保留原逐区计数匹配：同D29 fitting/root/C，每区搜索行数等于D30近期S，从该区全部原S日期label-blind抽取，三个搜索seed共18候选。重合度高、对比有限，未观察到差异即不确定；不设成功门槛、不自动选方案。D31后停止这六个目标月上的搜索窗口/门槛/C拆分变体，转向地图效用的下一步有界方案，在既有授权内交监督方科学评审，不自动进入Stage2/3。完整契约见[d31-matched-search-plan.md](d31-matched-search-plan.md)。

## D32：Stage1分配证据导出（已完成）

用户要求先解决Stage1 root工程问题，Stage2公式问题保留。新增每候选空间证据导出，预测路由与评分不变；不声称修复过拟合。契约见[d32-stage1-assignment-plan.md](d32-stage1-assignment-plan.md)。

## D33：浅层截断重放（已完成）

Stage1过拟合机制诊断：六个冻结D29候选的root/depth1/full同行比较，无新拟合、无选择。见[d33-shallow-replay-plan.md](d33-shallow-replay-plan.md)。

## D34：E1硬F1对Brier配对对照（已运行）

21根（7个剩余Stage1日期×H4/8/12）、42候选，仅E1质量不同；开发对照，非独立验证。见[d34-e1-brier-contrast-plan.md](d34-e1-brier-contrast-plan.md)。

## D35：全局+20轮容量对照（已批准）

21个D34 root各在原fitting池上全局续训一次L1 20轮，对照root/hard/Brier；不新建root、G、E1/E2。见[d35-global-increment-control-plan.md](d35-global-increment-control-plan.md)。

## D37：时间加权root对照（已完成，不采用）

仅改样本时间权重（半衰期24个月，均值1，仅fitting），21个加权root vs 原root vs persistence（E3为主，C诊断）。见[d37-recency-root-plan.md](d37-recency-root-plan.md)。下一候选的设计讨论（非实施批准）见[research/d38-root-next-options.md](research/d38-root-next-options.md)。

## D38：固定弱persistence初始margin的root对照（已完成；探索性候选，不采用为默认）

仅把四类均匀初始改为固定λ=.5的精确起点persistence margin（缺失起点=原默认.5），21个anchored root vs 原root vs 固定prior-only vs 无拟合post-hoc控制 vs persistence（E3为主，C诊断）；不训练局部模型。零拟合前置检查见[research/d38-prior-support-findings.md](research/d38-prior-support-findings.md)。见[d38-persistence-margin-root-plan.md](d38-persistence-margin-root-plan.md)。

## D39：现存概率的排序、校准与决策诊断（已完成，仅诊断）

仅用D38已核验E3行，无拟合/阈值搜索/校准器。排序信号存在；固定mass>=.5劣于argmax；按H校准方向不同。见[d39-probability-diagnostic-plan.md](d39-probability-diagnostic-plan.md)与[research/d39-probability-findings.md](research/d39-probability-findings.md)。

## D40：起点前决策规则可行性对照（已完成，不采用）

同H同臂、真实目标月U<O的旧E3预测上确定单一危机质量阈值（精确F1最大、并列取最大tau，>=3日期），6折启用、15折回退argmax；不改终点或概率，不选臂。结果：全21两臂仍低于persistence，启用子集变化不稳定；不采用。见[d40-forward-decision-plan.md](d40-forward-decision-plan.md)与[research/d40-forward-decision-findings.md](research/d40-forward-decision-findings.md)。

## D41：冻结分区的局部增量减半对照（已完成；诊断候选，不采用）

D34的21个Brier候选，保存概率上的几何收缩`p_half∝sqrt(p_root·p_local)`（零增量行复制root），C/E3同键对比root/full/half，另报真实局部/零增量路由分层；无alpha网格、不改终点或地图。结果：E3转移弱且异质（H8 Brier变差），C为插值；不采用。见[d41-local-shrinkage-plan.md](d41-local-shrinkage-plan.md)与[research/d41-local-shrinkage-findings.md](research/d41-local-shrinkage-findings.md)。

## D42：旧地图与当前地图的共同重拟合对照（已完成；不采用）

12个目标：最新同H旧地图（U<O）与当前地图，均在当前root与当前fitting行上按空间成员各区续训L1一次（同支持/回退规则），对照root与D35 global20；C/E3同键，C仅描述。不证明学习几何优于任意分区。结果：E3效应弱且异质，各臂均低于persistence；不采用。见[d42-map-transfer-plan.md](d42-map-transfer-plan.md)与[research/d42-map-transfer-findings.md](research/d42-map-transfer-findings.md)。

## D43：时间块Brier地图学习与共同重拟合（已完成，不采用）

21组D34 H/T：最新三个实际标签月为S_tb，较早为FIT_tb，拟合时间块搜索root并做一次Brier/root/L1/gt0搜索；时间块地图与D34随机地图都在当前D34 root与FIT上共同重拟合，对照root与D35 global20，仅E3评分。先做H4/H8/H12 2018-06三次生产等价搜索。本轮后停止时间切分/地图生成变体并综合Stage1。结果：H12描述性F1上升但Brier变差，H4 F1下降，H8近平；各臂均未超过同键persistence；不采用。Stage1综合见[research/d43-temporal-map-findings.md](research/d43-temporal-map-findings.md)；下一步规划从pooled/root预测问题出发（研究重点，不改变最终主模型或规格）。见[d43-temporal-map-refit-plan.md](d43-temporal-map-refit-plan.md)。

## D44：零拟合full-pool vs r80-FIT root诊断（已完成，不采用）

15组H4/H8/H12×2019-02…2020-06：v1保存的59个月窗口full-pool全局root与D34 r80-FIT root（实际FIT约75%）在完全相同E3键上比较，30个保存模型原始重放、0拟合；全键与persistence同键队列；不归因于样本量，不改模型或规格。结果：同键F1 r80-FIT .566612、full-pool .562348、persistence .592605；不采用full-pool替换，无比例/seed网格；Stage1仍未解决。见[d44-full-pool-root-diagnostic-plan.md](d44-full-pool-root-diagnostic-plan.md)与[research/d44-full-pool-root-findings.md](research/d44-full-pool-root-findings.md)。

## D45：保存root的boosting前缀学习曲线诊断（已完成，未采用策略）

21个D34 root，在H4 50/100/200、H8/H12 100/200/400轮的同一拟合路径前缀上评估FIT（样本内）、C（窗口内插值）、E3（前向）；对数损失与危机Brier帮助解释容量，危机F1仍为科学目标；不选最佳轮数、无早停策略、0拟合。结果：一半→全部时21组FIT/C两种损失均改善，E3概率质量常变差而H8/H12危机F1上升；各H所有前缀均低于同键persistence；不采用策略、无轮数网格；Stage1仍未解决。见[d45-root-prefix-diagnostic-plan.md](d45-root-prefix-diagnostic-plan.md)与[research/d45-root-prefix-findings.md](research/d45-root-prefix-findings.md)。

## D46：2×危机类加权root训练对照（已完成，不采用）

保持危机F1终点。21个D34 pair，原root、仅用FIT标签2:1危机加权的新root、零拟合posthoc×2控制及同键persistence；主要对照weighted−original、weighted−posthoc、weighted−persistence；≤21次root拟合，无权重序列或自动采用。结果：同键F1 weighted−original在H8/H12上升、H4下降，各H均低于persistence；根内AUC/AP与未加权Brier/对数损失各H均变差；不采用，无权重序列；Stage1仍未解决。见[d46-crisis-weight-root-plan.md](d46-crisis-weight-root-plan.md)与[research/d46-crisis-weight-root-findings.md](research/d46-crisis-weight-root-findings.md)。

## D47：深度1（stump）root对照（已完成，不采用）

危机F1仍为主要终点。21个D34 pair，仅把复制的G配置max_depth改为1，其余（FIT、种子、轮数、eta、正则、子采样、162特征、四类目标）不变；对照原root与同键persistence；根内AUC/AP为次要机理诊断；≤21次拟合，无深度/轮数搜索或默认采用。结果：E3同键F1各H均低于原root与persistence；不采用，停止深度/轮数序列；Stage1仍未解决。见[d47-stump-root-plan.md](d47-stump-root-plan.md)与[research/d47-stump-root-findings.md](research/d47-stump-root-findings.md)。

## D48：已知起点FIT限制对照（已完成，不采用）

危机F1为主要终点。21个D34 pair，仅把原FIT限制为hist_phase_o00有限的行（不用目标结局值选行，仅依据原有标注FIT池中已知起点标签特征的可得性；不替代、不加权），新root与原root及persistence在起点已知的FIT/C/E3键上比较；缺失起点E3只报排除计数；样本量、日历制度、构成等同时变化，非缺失机制因果检验；≤21次拟合，不采用。结果：起点已知E3危机F1各H均低于原root与persistence（FIT/C已知键改善）；不采用，不再继续时代/可得性样本池序列；Stage1仍未解决，不自动开始D49。见[d48-known-origin-fit-plan.md](d48-known-origin-fit-plan.md)与[research/d48-known-origin-fit-findings.md](research/d48-known-origin-fit-findings.md)。

## D49：零拟合排序余量诊断（已完成，不采用策略）

危机F1主要终点与最终标准不变。仅读21个已核验D38 E3保存行（哈希绑定D39/D40），起点已知键；对原root与anchored臂的s=(p2+p3)/Σp做确定性s≥c阈值族的事后（用E3真值）最优F1、persistence弱支配检查及persistence固定预测数k_p的整块括号点。仅为该冻结检查点、该暴露折上标量阈值族的事后上包络；非四类/特征策略上界、非可部署规则、非验证或采用；无拟合、无新决策策略。见[d49-ranking-headroom-plan.md](d49-ranking-headroom-plan.md)。结果：事后（E3真值）原root最优标量阈值的逐折均值F1比persistence高+.011881/+.031974/+.017748（H4/H8/H12），胜出折6/7、5/7、3/7，弱支配仅3/7、3/7、1/7；在persistence预测数下平均TP差−19.429/−1.571/−21，事后余量不等于排序普遍优于persistence；不采用策略，不声称过拟合已解决；逐折均值与此前汇总persistence仅因聚合方式不同；Stage1仍未解决，无D50。见[research/d49-ranking-headroom-findings.md](research/d49-ranking-headroom-findings.md)。

## D50：精确起点阶段排序诊断（已完成，不采用策略）

危机F1主要终点与最终标准不变。同21个哈希绑定D38 E3文件、起点已知键（112,795行，排除713行）；按精确persistence_code 0/1/2/3分格，原root与anchored臂在D49归一化分数上的格内危机AUC（并列半分，P·N=0时为null并注明原因）、固定argmax混淆并与D49逐root匹配混淆加总一致；逐日期全部格与逐H有效折均值（n_valid/7）；无汇总AUC/阈值/国家切片、无拟合或策略、无数值成功阈值或采用。格内AUC>.5仅为超出恒定精确阶段信息的描述性排序。见[d50-origin-phase-ranking-plan.md](d50-origin-phase-ranking-plan.md)。结果：原root在精确阶段2（发生）内逐折均值AUC .655702/.598012/.605804（>.5为7/7、4/7、7/7），阶段3（缓解）.736468/.745844/.728402（各H均7/7）；否定“任何状态内都无排序”，但不识别因果特征贡献、不证明可迁移、不支持按日期路由、不解决过拟合；H8发生排序不稳定，同分数的严格递增的标量校准无法修复排序不稳定；阶段1与阶段4或5支持度小，不称稳健；不采用策略，Stage1仍未解决，无D51。见[research/d50-origin-phase-ranking-findings.md](research/d50-origin-phase-ranking-findings.md)。

## D51：历史+日历78特征root消融（已完成，不采用）

危机F1主要终点不变。21个D34 pair，原FIT行/顺序/标签（含缺失起点行）不变，仅把新root的矩阵按schema块投影为75个history_blocks+3个known_calendar（含target_year），共同移除静态28+动态41+遗留15=84个特征；G/轮数/colsample冻结（特征维度改变有效搜索，已披露）；对照全162原root与精确起点persistence；主要读E3匹配键危机F1（对原root与persistence），附全键、Brier/对数损失、AUC/AP与D50格内AUC；联合移除，不能归因于单一变量/块或因果过拟合机制；≤21次拟合，不采用、不外推分区可行性。见[d51-history-calendar-root-plan.md](d51-history-calendar-root-plan.md)。结果：匹配E3汇总危机F1原/h78/persistence为H4 .629050/.633073/.651697、H8 .533375/.557674/.555614、H12 .478355/.512267/.549808；H8汇总高于persistence +.002060但逐折均值低于persistence（.545162 vs .555568），仅3/7折胜原root；各H Brier/对数损失/AP与阶段3 AUC均变差，FIT/C F1下降；F1上升伴随危机预测占比上升；联合移除，不作变量归因，既不能得出“协变量都是噪声”也不能得出“仅历史即可解决”；不采用为默认，不自动开展特征子集序列；Stage1仍未解决，无D52。见[research/d51-history-calendar-root-findings.md](research/d51-history-calendar-root-findings.md)。

## D52：独立二分类目标root诊断（已完成，不采用）

提案（1108cc9）曾待用户决定；2026-10-02用户回答“可以”，批准在固定21个D34 root上各做一次独立`binary:logistic`（原四分类码≥2）诊断拟合，不改变四分类主合同（D26/R11仍有效）。四分类包、schema、模型、概率、终点与最终标准不变；不采用、不进入Stage2/3；真实拟合仍须规划提交、原生实现/检查、生产提交、监督方代码审阅及单独放行；原root重放与D49/D50一致性须在任何真实二分类拟合前通过。见[d52-binary-root-proposal.md](d52-binary-root-proposal.md)。结果：匹配E3危机F1二分类/原归一化质量/原argmax/persistence为H4 .601942/.611056/.629050/.651697、H8 .516022/.518645/.533375/.555614、H12 .465093/.470746/.478355/.549808，汇总与逐折均值均在各H低于原两种规则与persistence；其他指标不一（H4 Brier/对数损失/AUC/AP略好，H8 Brier/对数损失变差、AUC略低AP略高，H12均变差）；二分类FIT/C F1更高而E3下降，历史到前向差距扩大；容量与Hessian差异及在四分类下选的G使其不能纯归因于目标函数；不采用，不证明分区思路不可行；四分类主合同不变；Stage1仍未解决，无D53、无二分类调参网格。见[research/d52-binary-root-findings.md](research/d52-binary-root-findings.md)。

## D53：精确起点状态概率水平转移诊断（已批准）

零拟合描述性诊断：只读D52保存的63个rows文件（哈希对D52 completion），按root×角色（FIT/C/E3）×精确起点状态（0–3，缺失单列）报告观测危机率、两种分数（s_original、p_binary）的均值、偏差与Brier，及逐标签月份明细和E3−FIT、E3−C对比；角色构成不同，不能分离因果时间漂移或纯过拟合；不新建转移参照、不做可部署校正。见[d53-state-probability-transfer-plan.md](d53-state-probability-transfer-plan.md)。
