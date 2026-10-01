# Evaluation 分层契约 v1.0

2026-10-01；设计按D24采用，D25已授权执行。母包事实与新设计分开记录；本契约的研究证据来自规划阶段，不是已完成的新模型运行。
D8 批准评价分层，D10 批准 E1/E2 复用及选择偏差取舍，D13 固定 Stage 3 每区 global+local 相对 global 的 >0.01 启用门槛。D9 修订为固定59个月，D11 接受局部缺少“4或5”。
**D20/D23/D24当前状态**：Stage 1区域内随机80/20、50/50取代先前内部时间方案；D23将D12扩展为严格 >0 与 >0.01 两族。E1/E2复用、E3/E4外部评分、Stage 3时间拟合与D18条件地图筛选保持。seed、容量、支持及日程已按D24采用，统一见 `experiment-plan.md`；全开发期选择G造成的开发偏差另行披露。
以下源码路径用 P 表示 `FEWSNETFourClassBaseline/`，不能用根目录 legacy GeoRF/XGB 替代。

## 当前母包：用途分开，但部分数据复用

| 层 | 用途与比较对象 | 当前数据 | 当前规则 |
|---|---|---|---|
| E1 区域误差与 q/scan | 当前父模型在哪些区域集中出错，提出候选区域集合 | 当前父分支 `X_set==1`；父模型只拟合 `X_set==0` | 按区域/类别 TP、FP、FN 构造归一化质量，扫描更新 q 和区域 gscore；不是显著性检验 |
| E2 子模型接纳 | 新子模型组合是否优于当前父模型 | **复用 E1 的验证行**；子模型在各区域对应 `X_set==0` 拟合 | 汇总完整父验证集 confusion 后计算 fixed-four macro-F1，严格 gain >0.01；父模型赢平局；不是 p 值 |
| E3 Stage 1 目标月评分 | 整个已学习分区模型相对 pooled baseline 的月度表现 | 窗口外目标月 T；独立于该候选的内部验证行 | partitioned 与 pooled 的同键 macro-F1；pooled 在相同真实 fitting 行另拟合 eval 模型，不直接读取 root checkpoint |
| E4 Stage 2 consensus 权重 | 候选分区对最终地图贡献多少 | 消费 E3 的月度分数；无新预测评价集 | w=max(0,logit(clip(F_part))-logit(clip(F_pool)))，clip=[1e-6,1-1e-6]；无显著性检验 |
| E5 Stage 3 预测与最终统计 | 固定地图后的 pooled/local、persistence、expert 表现 | 每个预测起点前的滚动历史拟合；目标月逐键预测用于报告 | 当前 local 只按 ≥50 行/≥2 类资格拟合，无 local 胜过 pooled 的验证门槛；最终用配对国家 bootstrap 给出差值区间 |

现有拟合标签窗口为 `[O-35,O)`，内部每区随机留出约 20%，并非内部时间验证。
当前 Stage 3 使用该历史池全部真实行拟合，没有 E1/E2 式内部验证。

## E1：q 的含义

对当前父分支的验证预测，区域 g、类别 k 的整数 TP/FP/FN 给出：

    D_gk = 2 TP_gk + FP_gk + FN_gk
    D_k  = sum_g D_gk
    Y_gk = D_gk / (4 D_k)
    A_gk = 2 TP_gk / (4 D_k)
    C_gk = Y_gk - A_gk
    B_gk = (sum_g C_gk) * Y_gk / (sum_g Y_gk)

D_k=0 时 Y/A 对应列为 0。C 是观测错误质量，B 按父分支类级错误比例及区域 exposure 分配期望错误质量。
候选区域集合 S 上，扫描迭代更新：

    q_k = sum_{g in S} C_gk / sum_{g in S} B_gk
    gscore_g = sum_k [C_gk log(q_k) + B_gk (1-q_k)]

q 是按类别的错误质量倍率，用于寻找候选区域；四分类最终 q 为四个值，区域级数值是 gscore。
非退化情况下 q>1 表示该集合错误相对期望更集中；分母无支持或 q=0 时实现令 q=1。
初始化有区域×类别 C/B 比值，但不能把它与最终集合级 q、macro-F1 增益或 p-value 混为一谈。

源码：`P/src/metrics/fourclass.py:139-153`（scan_masses）、
`P/src/partition/partition_opt.py:207-212`（get_c_b）、`:942-1008`（scan）。

## E2：当前父模型，不是每一层都比较全局根模型

`base_eval_using_merged_branch_data` 加载当前 `branch_id`；只有空字符串分支是 root。
更深层以当前父 checkpoint 为比较对象，父 checkpoint 可能继承更上层模型及变换。
两个新子模型分别拟合后，在完整且相同的父验证键上比较四种路由：

    parent / parent
    child  / parent
    parent / child
    child  / child

先合并两侧 confusion counts，再计算固定四类 macro-F1；不是各区域 F1 的平均。
只有 best_candidate - parent >0.01 才接受，保留父模型的一侧复制父 checkpoint。
变量 `sig=int(accepted)` 不代表统计显著性；四分类可达路径不使用 legacy 二分类 sig_test。
该 >0.01 是当前母包内部接纳规则，新设计由 D12 沿用后又经 D23 扩展为 >0 和 >0.01 两族；它们不等于 D3 最终增益 CI 门槛。

源码：`P/src/model/train_branch.py:10-17,31-50`；
`P/src/partition/transformation.py:278-305,677-680,721-752`；
`P/src/partition/partition_opt.py:854-873`（select_macro_children）。

## E3/E4 与 E5 的边界

当前母包Stage 1外部目标评分未参与候选内部切分，但决定consensus权重，属于地图学习。新设计按D24全开发期选G，E3另条件于该配置选择；只保证该目标未进入候选fitting/E1/E2行池，不能声称E3未影响任何模型选择。
不能拿它再次声称整个地图/配置已经在独立测试上胜出。
训练与地图合法性必须分别追踪：不仅要查 RF/XGB 训练标签截止，还要查候选评分标签截止。

最终 Stage 3 的逐键预测可用于用户已定的三 horizon 科学验收；D3 要求各自相对 persistence 的配对增益 CI 下界 >0。
该要求没有自动在每个节点启用 bootstrap，也没有把 q 或 consensus 权重转成假设检验。
历史 2021–2024 已被查看的限制继续披露，不能声称 untouched holdout。

源码：`P/app/main_model_GF.py:97-125,137-163`（实际 split 及目标月评分）；
`P/src/model/GeoRF.py:301-317`（真实 fitting pool）；
`P/scripts/step4_similarity_matrix.py:50-68`（compute_plan_weights）；
`P/scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:111-174`（fit_fold）；
`P/scripts/report_fourclass.py:120-138`（bootstrap）。

Stage 2 当前将 fs1–fs3 的完整候选账本共同送入 general consensus（`P/scripts/run_stage2.py:68-79`）。扩展 Stage 1 生成方式只增加有身份的候选来源，不自动改变共识算法或拆成每 H 独立地图。

D21区分候选诊断与最终目标：趋同可能是稳定性，高差异/数量不是验收，Stage 2不保证预测最优。方案取舍用开发期预测，最终Stage 3不调参。D24保留超过expert的H4/H8优化目标及增益/区间报告，不新增expert硬CI；persistence仍按D3/D17。

## 新设计：已确认契约

逐层区分目的/行池/决策对象。Stage 1随机fitting/validation来自外部O前合法池且键互斥，E3目标T不进入这两个行池；D24选择G可使用开发目标，故不声称T未影响任何候选配置。Stage 3内部global/local在各自V=U−H前合法拟合，不能用当前O模型回看U。Stage 1随机验证不声称满足每条验证行自身origin的前瞻约束。
按 D14，各内部 origin 的全局／父／子模型及其验证预测采用相同的 dense 输入、NaN 原生缺失处理与 ±inf→NaN 清洗，不拟合或跨 origin 复用 RF 填充值；前缀中已学得的缺失分支方向保持不变。特征定义和时间可用性仍按 D7 检查，原生缺失处理不授权改变历史统计公式。

按 D20，E1/q 与 E2/接纳复用同一候选的区域内随机验证行，父子模型只拟合该候选的 fitting 行；在候选外部 O 截止下利用 validation 标签寻找边界并选择子／父路由。D10 的复用原则保留，原多 origin 实现安排被取代。搜索与接纳仍是不同用途，复用不把 q 变成性能增益或显著性统计量。
由于边界／路由已看过该验证池，E2 只是内部筛选，存在选择偏差，不能宣称这些行是完整分区策略逐起点的独立前瞻重放。整个搜索／接纳流程另由外层时间评价检验。E3 仍保留窗口外目标月用于候选评分/consensus，因其参与地图开发，不能自动替代最终独立证据；最终 E5 不用于调参。
此前E1/E2严格顺序隔离及Stage 1内部多起点方案均已被取代。Stage 1 seeds/候选预算及Stage 3日程按D24采用，见experiment-plan.md。

候选接纳池在多个节点/配置中复用有选择偏差；D10接受该内部取舍，不取消容量限制与外层评价。支持/接纳规则现按D24固定，不足则按规则回退或报告，不临时改变行池。D11只排除“缺4或5本身即禁用”，其余支持资格保持。

增量轮数和回退属于选择过程，需注明开发行/有限预算；D24采用固定轮数，无early stopping。
按D15每H统一选G/L各一套，跨分区、分割层及最终folds复用；不逐区调参。不同区域的增量和D12/D23、D13路由仍可不同；有限网格及顺序选择按experiment-plan.md。
按 D9 修订，全局／父／子拟合标签窗都在自身 origin 取 `[origin-59,origin)`，内部 fitting 可以完整回溯到外部 fold 当前窗口之前；仍检查当时标签与特征可用性。窗口长度不再是开发搜索变量，不改变特征工程回溯公式。

每层日期/键、父与root/pooled身份、支持/回退及预测消费角色已在experiment-plan.md固定规则；实施时保存实际键和身份。只存总分不足以重建这些关系。

## 样本核查后的修订

详见 `research/sample-support.md`。Stage 1 每行政区当前随机 validation 只有 0–2 条；Stage 3 的 local fitting 虽有 1,016–7,728 行，却只涉及 8 个真实月份，40% 的分区×fold 无“4或5”。这些是现有 RF 地图与真实标签的计数，不是未来 XGB 分区表现的预测。

探索 E1=`[O-12,O-6)`、E2=`[O-6,O)` 实际只有 2／1 个观测月份；Stage 3 的 E2 缺“4或5”比例达 57.3%–60.7%。因此不能把两个日历上六个月的块当成两套充足的验证证据，本轮不冻结该拆分。

旧窗口诊断须区分两种统计：历史池内前缀 `[O-35,O-12-H)` 在 Stage 3 仅有 H4/H8/H12 的 4／3／2 个月份；内部起点完整回溯 35 个月可各有 8 个真实月份。前者不是数据源历史上限，也不是已修订的 59 个月新契约。D10 的每条内部预测记录父树、子树和输入表示的 fitting origin；边界／路由选择另记录开发截止身份，不冒充内部逐起点完整前瞻策略。

若 E2 声称是当时可运行的前瞻评价，E1 标签及选出的边界应在每个 E2 origin 已可用。E1 target 早于 E2 target、或两者行不相交，均不足以保证此点；上述相邻块尤其未证明 H8/H12 的搜索信息截止合法。确定具体日程时同时检查 selection cutoff 与 fitting cutoff。

W=59在Stage 3增至14个真实月份，不证明收益。D9取代35/59择窗；D10允许内部复用，D11接受局部缺稀有类，Stage 1按D12/D23、Stage 3按D13。D24已采用具体日程/支持规则；D3最终CI目标不变，未授权训练。

已确认 D12/D20/D23：E2 在完整父随机验证键上汇总两侧 confusion 后计算 fixed-four macro-F1；候选按 threshold_family 使用严格 gain >0 或 >0.01，等于自身门槛不接受，父模型赢平局。D23 只扩展 Stage 1 的两档固定生成对照，E3/E4 同口径评分，Stage 3 继续 D13 的 >0.01，不作用于 q 或 D3 最终 CI，不授权其他阈值搜索。

已确认 D13：Stage 3 的每区增量必须在该区相同历史时间验证键上，相对 global 的 fixed-four macro-F1 严格增益 >0.01 才启用；等于 0.01、更低或没有可用验证收益证据时零增量回退全局。每条验证预测的拟合标签须在其内部 origin 可用，所有路由选择所用标签须在本次外部 fold origin 可用；不能用本次最终目标标签择路，也不能以当前全局模型回头预测内部过去起点。
分区内合并各内部origin的confusion再算fixed-four macro-F1，不平均月F1或缩短类别轴。保存配对键、global身份、得分/回退原因；通过的路由用于外部fold重新拟合。该区总体不同于E2完整父分支，gate不是独立显著性结论，也不保证整体F1提高。D24固定最近六日期与至少三个有验证行且local可拟合日期；local失败日期保留global预测参与配对，D11仍适用。

**已确认 D18 的地图截止**：允许用外部 origin O 已知的地图作条件历史筛选，使用更早验证目标 U<O，每个内部模型仍只在 V=U−H 以前合法拟合。地图可能利用过 U 的标签，因此分数仅用于内部路由，不能称完整策略在 V 的独立前瞻重放。最终地图／超参数／阈值只用截至 2020-12 的开发信息，Stage 3 标签或成绩不参与这些选择；本次外部预测不得使用其 origin 尚不可用的信息。D16 已批准的已到达标签滚动拟合／固定规则筛选继续适用。D18 为此项单独确认，不混同 D10 的 E1/E2 复用。

## 已确认的开发与最终时间边界（D16）

按 D16，以 2018–2020 目标月作为开发／地图学习阶段，最终配置与地图选择的信息截止为 2020-12；之后按母包 H4=2021-05、H8=2021-09、H12=2022-01 起至 2024-12 滚动评价。该阶段边界已确认，但不替代内部 origin 的完整日程，也不自动证明既有 E3 是整体流程独立外层评价。
2021年后到达标签可在后续合法origin按59个月及D13更新，不据最终成绩改参数/阈值/地图。开发按origin截断候选评分目标后重建consensus，不用最终2020地图回测早期。D24先全开发期选G，因此开发地图/分数条件于配置选择，并非全部选择信息在历史origin已知；用户已接受且报告须披露。最终截止不放宽，已查看基线的历史仍称回溯评价。

## 已确认的最终置信区间（D17）

按 D17 沿母包使用国家块配对 bootstrap，对保存的最终同键预测进行重采样，不在每次抽样中重新拟合模型／选择地图。一次抽样对国家有放回抽取，与该国家关联的全部区域和整段月份同时保留；所有模型、cohorts 和 H 共用国家抽样次数。先按次数合并各国 confusion counts，再计算各模型 fixed-four macro-F1 及配对差值，不平均国家 F1，也不将每条区域月份当独立样本。

复用母包 `scripts/report_fourclass.py:28-30,96-138,152-167` 的 seed=42、2,000 次有效抽样、最多20,000次尝试、差值2.5%/97.5%分位数（linear）。某个必需 cohort 完全空的抽样重抽并记录；单个类别缺失不作为重抽理由，保持固定四类轴。有效抽样不足时区间证据为 incomplete，不能判 D3 通过。保存国家权重、逐次差值和配对键；D3 仍要求每个 H 的点增益及 CI 下界均 >0，不把三个边际区间说成同时95%区间。

解释限于这段已观察历史、固定预测下的国家块重采样，保留块内时间／空间依赖，不自动涵盖跨国共同冲击、未知未来年份或重新训练／选择的全部不确定性。已有按国家／年度的描述性表补充稳定性，不新增逐年胜出硬门槛。该方法与解释范围已按 D17 确认，不迁移成 E1/E2 或每区内部显著性检验。

## D26 修订（2026-10-01）：二分类危机终点

E1/E2/E3/E4 的得分统一改为 crisis-positive F1（IPC≥3=类码{2,3}；四分类 argmax 后折叠）：E1 用 crisis 单列 TP/FP/FN 质量；E2 为完整父验证键上精确 crisis F1 增益（>0 / >0.01 两族）；E3 为目标月 partitioned 与 root 的 crisis F1；E4 权重输入为该 crisis F1。四分类 fixed-four macro-F1 作为次要记录。Stage 3 gate、开发择优与最终 D3 的对齐见 experiment-plan.md A1，待 Stage 1 结果审阅后确认。
