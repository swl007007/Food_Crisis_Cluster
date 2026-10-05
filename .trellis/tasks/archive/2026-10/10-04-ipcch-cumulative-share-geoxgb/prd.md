# IPCCH cumulative-share GeoXGB architecture

Status: **v1.0 — science and execution contract frozen for baseline; P0 foundation validation and freeze precede scientific execution under R51.**
Date: 2026-10-04. This task was created with `--no-start`; the existing
`pooled-onset-confirmatory` task and its session pointer remain unchanged.

## Goal

Combine the completed local FEWS NET GeoXGB spatial architecture with IPCCH four cumulative-share regressors and a fixed 20 percent label decoder. Brainstorm/spec/grill precede infrastructure, freeze and Claude execution, with Codex supervision under R51.

Earlier requirement notes saying a decision did not authorize implementation record their historical approval boundary. The user's latest R51 instruction now authorizes the ordered infrastructure/freeze/handoff workflow; it does not change the accepted scientific contracts or authorize changes to reference packages.

## Requirements

- R1 用户确认：以本地完成包为基础，研究自适应统计驱动的空间分区与 XGB regression 的结合。
- R2 用户确认：新完成的 FEWS NET GeoXGB 包提供空间建模架构参考；本地 IPCCH GeoRF 包提供 IPCCH 接入参考。
- R3 用户确认：IPCCH 的核心决策参考为四个 small XGB regression → 固定 20% 门槛 → label。四个累计目标的代码依据见 research/architecture-evidence.md。
- R4 用户指定顺序：brainstorm → 落盘 spec → grill。此版记录建议和待定问题，不把建议视为确认。
- R5 保留原始数据、既有完成包与历史结果。新实验必须能与相同输入/拟合池的 pooled 四回归器基线比较。
- R6 明确区分连续累计 share、由 share 决定的 phase、官方 reported phase，以及分类概率；这些对象不能互相替代。
- R7 分区学习、局部模型采用与最终报告分别保留时间边界、评价样本和统计含义；首版无共识Stage2，见R34。
- R8 用户确认：以最终标签表现为主；报告四分类 accuracy、macro-F1，二分类 accuracy、F1、precision、recall、F2，以及连续 q3 的 R²。
- R9 用户确认内外层分工：四个 XGB 内部以 squared error 拟合累计人口 share；外层空间分区与模型配置按验证集危机正类 F1 选择，解码门槛固定 `>=0.20`。不引入直接优化硬 F1 或 soft-F1 的 custom objective。最终报告 R8 全部指标并与 persistence 比较；gate比较对象与增益门槛见R16，结构支持见R28，危机类别支持见R29。
- R10 用户确认：与 persistence 比较；每个行政区采用预测起点时合法可用的最近一期有效真实人口观测，将该期人口推导的 phase 和同一期 q3 延续到目标月，记录来源日期与年龄。无合法历史时 persistence=NA，保留模型预测；配对比较仅用双方有结果的相同 keys，同时报告覆盖率及模型全样本结果。源数据可用性采用R22的观测月末口径。
- R11 用户确认：分类评价真值采用真实人口占比推导的 phase。所有标签调优、分类指标与 label persistence 使用这一真值口径；reported phase 保留原始字段，但不作为本实验评价真值，也不能填补缺失的 share-derived truth。
- R12 用户确认：累计人口占比恰好等于 20% 算达到该 phase，使用 `>=0.20`；真实值和预测值一致。旧 IPCCH GeoRF 的严格 `>0.20` 标签不能原样复用。此确认不决定 rounding、占比分母、缺失或单调处理。
- R13 用户确认：保留四个回归器及五级原始 phase 解码；四分类评价合并为 1、2、3、4或5；二分类将 phase1–2 归为非危机、phase3–5 归为危机正类。二分类 F1、precision、recall、F2 针对危机正类计算。真实值、预测值和 persistence 使用同一类别映射。
- R14 用户确认：q2–q5 共用空间分区图；全局四模型作为基座，每个区域分别保留对应目标全局模型的树，再追加少量局部树。四模型作为完整组合共同采用或回退，局部改善不足时使用全局组合。跨horizon分区范围见R24，局部容量见R43；采用门槛见R16。
- R15 用户确认：四个原始回归预测先做最小平方改动的有界单调投影，满足 `1>=q2>=q3>=q4>=q5>=0`，再不经四舍五入按 `>=0.20` 解码；原始预测另存。该政策应用于模型预测，不授权修改原始人口数据或修补无效真值。q3 R²的主次口径见R25。
- R16 用户确认沿用两级经验增益 gate：Stage1 候选相对当前 parent 的危机 F1 严格增加 `>0` 才接受；Stage3 区域四模型组合相对同 fold 的 global 组合，在相同合法历史验证 keys 上的危机 F1 严格增加 `>0.01` 才采用，否则回退。0.01 为绝对 F1 增量；平局或恰好门槛不通过。两层另设支持要求；本方案不新增逐分裂显著性检验，不将经验 gate 称为统计显著性。
- R17 用户确认：首轮预测期采用实际月数 H={1,3,6,12}，origin O=T−H；不纳入 H0 nowcast。该决定仅固定预测期，不自动批准沿用旧包的样本、时间切分、特征、容量或既有分数。
- R18 用户确认：沿用 IPCCH GeoRF 完成包绑定的 `IPCCH_2026_completed.csv` 数据版本，覆盖其中全部国家和行政区；不按模型表现筛选国家。真实 share/phase 按本次规则重新构造，不直接复制旧二分类标签。源文件身份与路径见 design.md 第7节，实际样本数需按新契约重算。
- R19 用户确认：2014–2022为开发范围，分区、超参数及选择规则的信息截止为2022-12；2023–2025为冻结方案的主回顾性评价期，2026已有月份单独补充。所有主评价origin>=2023-01，H1/H3/H6/H12的首个目标月分别为2023-02/2023-04/2023-07/2024-01。评价期间按冻结规则用当时合法历史滚动重拟合；不据评价分数重新调参或改图。滚动拟合窗口见R23，Stage1内部切分见R35，历史gate日历见R37；最终预测折次与覆盖见R47。
- R20 用户确认沿用目标QC：P1–P4必须观测，仅缺失P5可补0并标记；各占比在[0,1]、人口为正；五项总和S在[0.90,1.10]（含端点）才接受，随后按P_k/S标准化，构造q2–q5和本次>=20%的phase。无效目标保持缺失、不得用reported phase填补；原始数据不覆盖。P5补0是明确建模约定，不是源事实。
- R21 用户确认采用 IPCCHPopulationHistoryExperiment 的 rich561 配方（original93 + 468历史特征），作为全局/局部四回归器共同输入。每行按自己的O=T−H构造，历史不足保留NaN并由XGB原生处理，不因缺少历史或persistence删行。依赖危机定义的历史特征按本次>=20%重新生成；不继承旧阈值搜索、correction模型或history-only拟合/评分限制。信息可用性口径见R22。
- R22 用户确认沿用回顾性观测月末可用口径：滚动预测origin O=T−H含当月月末，协变量、人口历史和训练标签的观测月份须<=O；每条历史训练样本的特征仍按其自身origin构造。persistence遵守同一截止。未核验历史发布日期及修订版本，不据此声称历史实时可得；滚动拟合窗口见R23。Stage1已按R35采用截至2022-12的pooled开发F/S，特征仍遵守各行origin，但内部S不声称严格前推预测验证。
- R23 用户确认：每次滚动预测使用36个日历月的训练目标窗口[O−35,O]，包含两端；global使用窗口内有效训练目标，local从同一拟合池按区域取子集。不以36次有效观测替代日历窗口，不因缺少标签自动向前扩窗。每行特征仍按自身origin构造；rich561历史及persistence可追溯窗口前的合法观测。Stage1按R35使用2014–2022 pooled开发池，不被此36月滚动窗口截断；局部支持见R28–R29，全局支持见R40。
- R24 用户确认：H=1、3、6、12分别独立学习空间分区图，每个H内q2–q5共图。各图仅用截止2022-12的合法开发信息确定，主评价和2026补充评价均冻结；滚动拟合不重新学习地图。按R34直接冻结各H学到的地图，不跨H聚合；内部切分见R35，递归搜索预算见R45，允许不同H最终得到相同或无分裂地图。
- R25 用户确认：q3 R²主表使用有界单调投影后的q_star3，补充报告投影前q_raw3的R²；二者在同一报告cohort使用相同真实q3及相同keys。与persistence的配对比较遵守R10，模型全样本与配对样本分别标记。不得根据评价分数交换主次；投影不保证单独改善q3 R²。汇总见R26，undefined指标规则见R27。
- R26 用户确认：主表按H分别汇集主评价期全部有效行政区×目标月预测，每条记录等权，统一计算全部指标（含q3 R²），不平均逐月指标或跨H混算。2026补充期单列；model/persistence按共同keys配对，全模型样本另列并报告覆盖率。逐月/逐国结果作诊断；主表不作人口加权或国家/月度等权再平均。此决定固定报告汇总；训练权重见R30，内部候选选择汇总见R44。
- R27 用户确认：无法定义的指标记NA并保存原因。precision/recall/F1/F2按各自公式分母判断，合法0保留；四分类固定1、2、3、4/5，某类在真值与预测均未出现则该类F1和四类macro-F1均NA，不删类重算。q3样本不足2或真值恒定时R²为NA，合法负R²保留。gate任一所需F1为NA则不接受分裂或局部组合；结构支持下限见R28，危机类别支持见R29。
- R28 用户确认采用局部模型结构支持下限：每次区域拟合>=500个有效原始行政区×目标月keys、>=50个行政区、>=6个不同目标月份；Stage1各候选区域验证>=100个keys、>=20个行政区、>=3个目标月份；Stage3区域历史验证合计同样满足100/20/3，并至少有3个历史验证日期成功拟合局部组合。每个H独立计数，四目标及重复视图不扩增支持，月份要求针对区域合计。支持不足时Stage1该区域保留parent、Stage3回退同fold global；这些是工程下限，不是统计充分性声明。危机类别支持见R29，常数目标处理见R31，全局支持见R40。
- R29 用户确认新增验证类别支持下限：在R28的100条验证记录等要求之外，真实危机与真实非危机各>=20个原始keys。Stage1逐拟采用候选区域检查；Stage3按区域跨历史验证日期合计检查，不要求逐月分别达标。类别按本次真实人口推导的phase>=3定义，不按模型预测计数；不足则保留parent或回退global，不删评价记录。这是新增工程下限，不是原包既有门槛或统计充分性证明，也不自动施加于回归拟合池。
- R30 用户确认：回归训练中每个有效行政区×目标月记录等权，四个目标及global/local采用相同政策。不额外按人口、危机类别或时间远近加权，不作类别重采样。各拟合池中每个原始key的样本权重为1，局部仅因区域成员资格取子集；内层保留squared error，外层仍按验证危机F1选择。配置选择见R44；按R34首版无共识权重。
- R31 用户确认：某个累计share目标在拟合池内恒定时，仍按统一XGB squared-error流程拟合，不因单目标恒定拒绝整个四模型组合。global照常拟合，local保留对应global prefix并追加局部树；记录每目标常数标记和值，不直接用常数预测器替代booster。既定样本支持及组合F1 gate仍须满足；有限轮局部修正不保证预测精确等于该常数。空拟合池和数值异常不属于常数目标政策。
- R32 用户确认采用单列危机F1误差扫描：parent四回归输出先按已定规则投影/解码，以真实/预测危机标签按空间group汇总TP、FP、FN，沿用最新GeoXGB以D_g=2TP_g+FP_g+FN_g、误差FP_g+FN_g构造的扫描统计。父区域F1无定义或总误差为0时无候选并保留parent；候选仍须通过结构/类别支持和危机F1严格增益gate。每个原始key计一次，TN保留在支持与评价中。扫描分数不是p值，连续share不作分类概率；具体公式见research/scan-statistic.md。
- R33 用户要求调查Stage2必要性及历史IPCCH候选分区支持。已核实旧IPCCHGeoRF明确采用一次pooled分区学习、无Stage2，稀疏验证/评分支持是历史讨论背景；新方案尚无已确定且有证据支持的同H多图候选池。此前logit共识权重提案未获确认，调查建议省去共识，用户随后按R34采用。证据见research/stage2-necessity.md。
- R34 用户确认首版省去共识Stage2，各H直接学习并冻结一张空间分区图，再执行滚动global/local四回归器、组合采用gate及统一投影/解码。不构建为共识而生成的月度/年度多图池，不运行多图加权相似度、k40/eigengap/谱聚类流程，不设置logit共识权重。保留递归分裂候选搜索和已确认的每H独立地图；未形成有效分区时使用同源global。Stage1内部切分见R35，未学习/未分配地区路由见R36，平滑和底图复用见R38–R39；数值容量及固定拟合参数见R43，配置评分见R44，递归预算见R45，候选规模细则见R46，拟合复用与总计算量见R48，软件环境见environment-lock.json，本决定不授权实现。
- R35 用户确认沿用IPCCH的Stage1区内时间各半切分：2014–2022有效原始行政区×目标月记录按各区时间排序，n>=2时最早floor(n/2)为F拟合、其余ceil(n/2)为S验证；四个H共用原始key分组，各自按自身origin构造特征并独立学习地图。n=0/1地区不参与分区拟合、扫描或split gate，但保留地理/预测覆盖及后续合法历史/训练资格。该切分仅保证区内时间先后，不保证跨区统一时间隔离；S分数用于开发搜索，不称严格前推预测成绩。最终表现仍由独立于地图选择的2023–2025滚动评价判断；未学习地区路由见R36。
- R36 用户确认：未参与分区学习或无法可靠分配分区的行政区，直接使用同一H、同一fold的global四模型，不做最近邻/100km donor分区补全。保留预测及满足真值条件的评价记录，逐行记录回退原因，报告实际学习/局部采用/global回退覆盖率。其合法人口历史仍可用于自身origin特征，合法监督记录仍可进入Stage3 global的同源36月训练池，不因缺少分区ID排除；Stage1 n=0/1排除规则仍按R35。无有效分区时与同源pooled四模型完全一致。
- R37 用户确认：当前(H,O)的Stage3历史gate统一选择全体有效人口目标台账中U<O的最近最多6个不同观测目标月份，不要求连续，不按区域挑选或为凑支持追加更早月份。每个U以V=U−H回放，训练目标窗口[V−35,V]，各训练行特征仍按自身origin，预测U用V时可用特征。区域跨日期合并混淆计数计算危机F1，不平均月度F1；沿用100/20/3、正负各20、至少3个成功且支持合格的局部拟合日期、NA和严格>.01门槛。局部支持不足的日期保留keys并使用该fold global，不计成功局部拟合日期；总支持不足则当前global回退。U不额外限定在当前36月拟合窗口。回放使用当前冻结地图/配置，它们可能利用了晚于历史V/U的开发信息，因此仅是给定地图/配置的历史采用gate，不是整个流程在V时已可实施的独立验证；当前目标T真值不参与。全局支持及空池停止见R40，技术错误处理见R41。
- R38 用户确认首版采用3轮共享边界邻接平滑及连通性诊断，不强制每个分区是单一连通分量。沿用同步邻居加自身投票，当前标签占比严格<4/9时切到另一候选标签；孤立或无有效邻居地区保持候选归属，未学习地区仍按R36回退global。平滑后再应用既定结构/类别支持与完整路由F1 gate，不在接受后强拆分量、改归属或用距离边强行连接。允许一个学习分区包含多个不相连片区，保存每个分区在学习成员诱导图上的连通分量及孤立地区数量，不宣称严格地域连通。底图版本复用及上游行政区匹配来源限制见R39和research/geometry-connectivity.md。
- R39 用户确认沿用IPCCH GeoRF完成包`ipcch-v1-20260920d/geography/`的修复底图、地区ID映射和共享边界邻接图，以完整组件哈希冻结身份，执行前核验文件、keys及缓存一致性。保留原跨国邻接，不增加国家屏障、不重做行政区匹配或几何修复；旧学习分区和donor归属不继承。明确保留已知数据限制：上游构造允许最近邻匹配，逐区匹配来源和边界vintage尚未完全核验，拓扑修复不证明行政区对应关系正确。该来源限制不豁免文件身份不符、key冲突或损坏错误；本决定不授权实现或修改原始/完成包文件。

- R40 用户确认：Stage1 global/root及Stage3每个必需的历史/当前global四模型组合，只要合法QC后原始key拟合池非空，就按已定回归流程拟合；不套用local的500/50/6或危机正负类别下限。记录拟合keys、地区数、不同目标月份数、危机类别组成及每目标常数标记，非空仅表示操作资格、不代表统计充分。必需的global拟合池为空时，记录stage/H/origin/窗口/原因并停止该运行、标记未完成；不得扩窗、删除gate日期或评价fold、调用未来模型或以persistence冒充模型预测。local及验证支持门槛保持不变；数值异常见R41，最终预测折次与覆盖见R47。

- R41 用户确认：正常局部支持/gain不足或预期未分配地区按既定parent/global回退；技术错误停止受影响运行并标记未完成。适用于Stage1搜索/配置评价及Stage3历史/当前拟合：任一已尝试的global/local拟合或预测抛错、任一原始q预测NaN/Inf、输出shape/key错位、投影失败、global prefix保留约束被破坏、必需模型/schema/地图产物损坏或身份冲突均须记录定位信息并停止；局部模型技术错误也不能自动global回退。不得按目标混搭、用常数填补坏预测、删记录/日期/候选继续评分，或更换种子/容量/窗口试到成功。合法特征NaN、undefined指标NA、常数目标拟合及有限越界/不单调预测的既定投影政策保持有效；详见research/error-policy.md。

- R42 用户确认：每个H独立选择一套global超参数配方G_H和一套local增量配方L_H；同H的q2–q5全局回归器共用G_H，四目标及各学习区域的局部增量共用L_H。Stage1与Stage3历史/当前拟合沿用冻结配方，不按目标、区域或日期另行调参；不同H可以选不同配方，选择只使用开发期危机F1。共享的是超参数，四回归器、区域增量及各fold重拟合仍独立，各目标仅继承对应global prefix，不共享跨目标树或叶权重。数值容量及固定拟合参数见R43，配置评分见R44，递归预算见R45，候选规模细则见R46，拟合复用与总计算量见R48，软件环境见environment-lock.json。

- R43 用户确认首版数值候选表：global深度{3,4}×{200,400}轮形成G1–G4，local为L1=(深度1,追加20轮)、L2=(深度2,追加40轮)，每H有8个G/L配方对、四H共32个配方对，不等于总拟合次数。共同eta=.05、seed42、固定轮数且不早停；global的min_child_weight/lambda/alpha=10/10/0、subsample/colsample_bytree=.8/.8，local为20/20/1及1/1。global显式base_score=.5，local保留对应全局基分和树再追加，不累计祖先local增量。四模型仍为独立squared-error回归，数值表与其他固定复现项见research/model-capacity.md。配置评分/平局规则见R44；递归预算见R45，候选规模细则见R46，拟合复用与总计算量见R48，软件环境见environment-lock.json，不授权训练。

- R44 用户确认：每H八个G/L配方使用同一2014–2022原始key F/S及同一R45递归搜索预算；完成各自Stage1后，在完整共同S上用最终已接受路由、统一投影/解码及合并TP/FP/FN计算危机F1，选择绝对F1最高者，不按相对各自global的增益排名。精确同分依次优先终端学习分区更少、G+L名义轮数更少、global深度更浅、local深度更浅、固定G/L编号更小；无分裂候选正常参选，NA不填0，全NA不冻结赢家，技术失败不准删候选后宣布完成。直接冻结获胜地图及配方，不共识、不以F+S再学习Stage1地图；Stage3仍合法滚动重拟合，matched pooled对照使用该fold同一获选G的global组合。S重复用于分区搜索与配置选择，仅是偏乐观内部开发成绩，不是独立前推验证或完整Stage3采用策略的直接优化；首版不增加滚动开发筛选层。详细证据及保存项见research/configuration-selection.md；递归预算见R45，候选规模细则见R46，拟合复用与总计算量见R48，软件环境见environment-lock.json，本决定不授权训练。

- R45 用户确认统一递归搜索预算：root成员深度0，最高child成员深度4，仅搜索parent深度0–3，最多16个终端成员分区；所有H/G/L使用同一上限，不再叠加源码80累计选择轮数cap。每个合格parent最多一次确定性单列scan初始化及固定1000轮迭代，采用最后候选，无随机重启或拒绝后重试；随后执行3轮平滑、既定支持检查、每个合格child最多一次四模型拟合及完整路由严格F1增益gate，只有接受的分裂继续递归。继承parent预测的child也占一个成员深度，不强制产生分裂。成员深度、XGB树深度及实际20/40追加轮数分开计数；每配置至多15次scan、30组local四模型即120次局部单目标拟合，global和Stage3另计。候选规模/排序同分细则见R46，Stage3及总实验拟合复用/计算量见R48；证据及条件上界见research/recursive-search-budget.md。本决定不授权实现或训练。

- R46 用户确认扫描候选规模规则：当前parent中不同学习行政区group计N（含零scan mass的S成员），平滑前两侧各占45%–55%，排序前缀m取[max(1,ceil(9N/20)),min(N−1,floor(11N/20))]的全部整数；无可行整数则保留parent、不扩范围。按分数降序、精确同分按冻结area-ID顺序；选允许范围内前缀分数和最大者，同分优先更接近一半，再选较小m。单列c/b初始化亦用相同分组函数；每轮规模比较不额外拟合child，仍按R45取1000轮后的最后候选。3轮平滑后不强行恢复45%–55%，记录前后规模并检查既定支持/F1；空侧不构成分裂。不按人口、行数或国家平衡，不逐字继承旧flex索引边界。该工程限制可能错过很小的高误差片区；细则见research/scan-candidate-size.md。本决定不授权实现或训练。

- R47 用户确认滚动预测日历与覆盖：主评价逐月列出全部122个H×目标月fold，H1/H3/H6/H12分别35/33/30/24个，各fold预测当月全部通过本次人口QC的原始行政区keys，不因缺历史、缺persistence或未分配地图删行。整月无有效目标时记录no_valid_target、n=0、指标NA，不请求该fold当前或历史gate拟合；非空预测fold中必需global空池仍按R40停止。2026仅运行冻结源内有有效人口目标的月份、各月四个H，单独报告并保存全年月份覆盖台账。不为无真值行政区×月份额外生成预测，不补truth，完整QC/地理/scaffold台账仍保留；结果是有观测真值样本上的回顾性表现，不声称全地区逐月预测覆盖。当前T的truth只定义评价cohort并用于最终评分，不进入特征、拟合、gate、路由或配置选择。详细日期、空月/缺历史区别及源码依据见research/prediction-schedule.md。本决定不授权实现或训练。

- R48 用户确认相同拟合复用、gate逐origin重算：新实验冻结数据/代码/环境内，完整身份一致的模型仅拟合一次、复用已保存模型。Stage1同H/G/F的global四模型可供L1/L2独立搜索共用；Stage3相同历史/当前global及local拟合可复用，不跨q目标、H、Stage1/3或旧完成包。身份覆盖origin/窗口或F成员、有序拟合keys、X/y/权重、schema、参数/种子/环境；local另绑定区域成员和对应global prefix，追加从独立副本开始。每个当前O仍按合法日期和keys重算支持/采用gate，不沿用旧采用结论或以已路由结果冒充原始local预测。缺缓存合法拟合，缓存损坏/身份冲突按R41停止；四模型完整路由政策不变。分别保存唯一拟合、请求/复用及用途证据。固定科学运行范围的保守上界为Stage1 3904次、Stage3 80920次单目标拟合，合计84824次，实际受复用/支持/空fold减少；不是时长估计或训练授权，不增加候选或截断后称完整。详细身份和计数契约见research/fit-reuse-budget.md。

- R49 用户确认最终比较与不确定性报告：逐H在E_all共同keys上比较GeoXGB与R44的matched pooled、在E_persist上与persistence配对，报告全部指定指标/差值和覆盖。仅主评价期两项Δ危机F1给95%国家整簇bootstrap描述性区间：每H/cohort seed42、固定2000次有放回抽国家，整带该国地区/月记录，比较双方共用国家倍数，合并TP/FP/FN后算差值，不跨H或平均国/月F1。至少2国、点差值可定义且全部2000次差值有限才报告2.5/97.5线性百分位，否则区间NA并留原因/抽样，不补抽、不填0或删坏draw后算区间。2026与国/月诊断仅点估计/差值/覆盖；其他指定指标不省略。区间条件于保存预测，仅描述国家构成变化，不含训练/分区、跨国共同冲击或未来外推不确定性，无多重比较校正。不要求全部H/国家/月战胜persistence才能交付，正负结果均如实保留，技术失败/必需证据缺失仍不能称完成；不得根据主期结果增加搜索。详情见research/comparison-uncertainty.md。本决定不授权实现或训练。

- R50 用户确认独立包方案：在当前仓库新建IPCCHGeoXGBExperiment，以IPCCH GeoRF完成包为数据/QC/日历/证据基座，所需GeoXGB及rich561代码作为可追溯固定副本移入新包并适配；运行时不动态依赖旧实验包源码或旧GeoRF ZIP后端。原始数据和R39已冻结几何保留只读外部输入，新模型/缓存/结果使用独立run目录，不修改既有完成包。细则见research/package-boundary.md。

- R51 用户授权：做好基建后冻结，然后handoff给Claude执行，Codex转为监督。先补齐环境/输入及来源清单、执行计划、上下文与验收契约；Claude在真实绑定会话和合法审计start后完成独立包基建，Codex核验其证据并固定基建提交，再交由同一Claude按冻结方案完成科学实现、测试、正式实验和交付。保留原任务/参考包，不伪造基线或审计通过；按既有controller流程close并核查结果。遇科学契约冲突、源身份不符或必需证据缺失时显式报告，不自行调参、补数据或宣称完成。运行时长不预先保证，阶段状态和确切下一步记录在本任务进度台账。

## Acceptance Criteria

- [ ] A1 三套参考实现及版本边界有可追溯文件证据；不把归档状态当作完整独立审计通过。
- [ ] A2 确定四目标、20% 边界、精度/越界/单调处理、truth 来源和缺失策略。
- [ ] A3 落实已确认的危机F1扫描、split/local gate与无Stage2直接地图路径；区分经验增益门槛和统计显著性。
- [ ] A4 冻结国家/时段/horizon、特征基座、时间切分与样本支持定义。
- [ ] A5 明确 pooled/local 相同数据契约、原始 key 计数、无分区及样本不足回退与证据输出。
- [ ] A6 Grill 后完成最终 spec 与执行计划；R51已授权，按implement.md先完成P0并经Codex核验冻结，再进入科学实现/执行。
- [ ] A7 R8 的完整指标面板及 matched persistence 比较可由保存的逐行预测重算；q3 R² 不混用危机分类概率。

## Execution boundaries after R51

R51授权按冻结执行计划开展本任务基建、实现、既定搜索/训练、验证、Claude交接及审计生命周期。修改其他任务、修复/覆盖原始数据、移动/替换参考完成包、超出冻结候选/日历的额外搜索仍不在范围内。
首版模型设计不包括共识Stage2、其多图候选池及权重/谱聚类流程。
首版不做最近邻或donor分区补全；未学习/未分配地区按R36直接global回退。
首版不施加严格分区连通约束，不以强拆分量或距离补边替代已确认的平滑和gate。
不自动引入 FEWS NET interruption augmentation、旧 162 特征、H4/H8、59 月窗口或旧支持阈值。

## Blocking decision inventory

1. 科学决策已闭合；准确环境、输入/源码身份、执行计划和上下文见本任务JSON清单及implement.md。随后固定实际审计起点与Claude执行身份；R50/R51已生效，不重复征求同一许可。
2. 准备阶段核验源身份、支持及请求计数；它们是执行证据，不是额外调参自由。未完成的基建检查不能被文件存在或handoff消息代替。

## Notes

- Keep `prd.md` focused on requirements, constraints, and acceptance criteria.
- Lightweight tasks can remain PRD-only.
- For complex tasks, add `design.md` for technical design and `implement.md` for execution planning before `task.py start`.
