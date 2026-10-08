# IPCCH 两套 pooled XGB 基线的区别

2026-10-07。比较对象是本仓库 MLP 实验引用的 **GeoXGB P6 pooled 基线**（`p6-formal-20261004b`），与相邻 `../IPCCH` 仓库 **10月6日完成的 `origin_safe_climate_idp_v1` global 实验**，不是10月5日的 `climate2015_v1`。

两者共享“跨地区 pooled 拟合 → 四个独立累计份额 XGB 回归器 → 20% 阈值决定 phase”的基本结构。这里的 ensemble 不是多种子平均。但两者并非只更换了特征。

| 项目 | 本仓库 P6 pooled XGB | `../IPCCH` 最新 global XGB |
| --- | --- | --- |
| 特征数量 | 各 H 均561列：original93＋人口份额历史468 | H0/3/6：无历史863、加 safe history 868、再加 IDP 870；H12：647/652/654 |
| 历史特征 | q2–q5、分布指标、最近6次观测、间隔、变化、窗口统计与危机事件等 | 最近3次 **reported overall_phase**，加最近一次与前两次的差值，共5列 |
| 气候及其他特征 | 原始天气、植被、冲突、价格、静态变量及较少衍生；特征数量主要来自历史块 | 14种月度/生长季气候指标的滞后、趋势、交互、空间等衍生，加其他深度特征；IDP臂另加国家级stock和报告年龄 |
| 回归目标 | 五个份额归一化后构造 q2–q5 | 同样归一化后构造四个累计份额；有效样本筛选规则不同 |
| **分类真值** | **按人口份额及20%规则推导 phase** | **报告的 overall_phase** |
| 预测解码 | 先有界递减 isotonic 投影，再对未舍入值用 `>=0.20` | 对四个未舍入原始回归分数直接用 `>=0.20`，没有投影 |
| 训练协议 | 每月重训，最近36个月，等权重 | 每个目标年拟合一次，使用年初安全截止日前全部历史，24个月半衰期加权 |
| XGB容量 | 深度3–4、200/400轮、学习率0.05；各目标共用所选G配方 | q3深度9，其他目标深度11；200棵、学习率0.1；另有采样与正则参数差异 |
| 期限与评价队列 | H1/3/6/12；主期2023–2025，起点随H变化；2026另报 | H0/3/6/12；2022–2025，四个H和三个臂均28,205个评价键 |

P6还要求人口有限且为正、单项份额在[0,1]、份额总和在[0.90,1.10]；最新global的份额有效性规则主要是五项有限非负、总和大于零，并要求有效reported phase。因此，输入面板、历史观测池和可评价样本也不同。P6的pooled配方是所选空间模型的匹配全局配方，并非单独优化的pooled最优模型。

**解释边界：** 最新global加强了外生特征，但历史份额信息反而少于rich561，不能简单理解成“在同一模型上增加特征”。真值、解码、训练窗口、权重、树容量及评价队列同时变化，当前分数差不能归因于特征设计。特别是分类真值不同，即使相同预测也可能得到不同F1。

MLP结果应表述为：**本次固定配方的MLP弱于P6的matched pooled XGB**。这不是与最新global XGB的同条件比较，也不能据此判断哪套特征更好。若后续要隔离特征贡献，应先统一真值、QC与评价键、期限、训练协议、解码和模型配方，再做特征替换。

## 后续：只移植训练协议的检验（2026-10-07）

按上述边界，只把相邻仓库的“每年拟合一次、全部历史、24 个月半衰期加权”协议移植到 P6（特征、真值、投影解码、模型配方和地图不变）。主期年度 pooled 相对 P6 pooled 的危机 F1 差为 H1 −0.0256、H3 +0.0032、H6 −0.0042、H12 +0.0012；年度 GeoXGB 相对年度 pooled 的点估计约在 ±0.0004，区间均含 0。即在 P6 设定下，这一训练协议本身没有带来稳定的 pooled 提升；两套 global 实验之间的分数差尚未逐项分解，其余差异（真值、解码、特征、树容量、评价队列）仍未隔离。详见 [结果说明](../../.trellis/tasks/10-07-ipcch-fixed-map-yearly-xgb/results.md)。

## 依据

- [P6冻结配置](../../IPCCHGeoXGBExperiment/config/experiment-contract.json)：目标、日历、窗口、参数及权重。
- [P6目标与QC](../../IPCCHGeoXGBExperiment/ipcch_geoxgb/targets.py)、[特征构造](../../IPCCHGeoXGBExperiment/ipcch_geoxgb/features.py)、[投影解码](../../IPCCHGeoXGBExperiment/ipcch_geoxgb/projection.py)。
- [最新global设计](../../../IPCCH/.trellis/tasks/archive/2026-10/10-06-global-origin-safe-climate-idp/design.md)、[运行证据](../../../IPCCH/.trellis/tasks/archive/2026-10/10-06-global-origin-safe-climate-idp/evidence.md)。
- [最新global目标、权重与解码](../../../IPCCH/src/ipcch/origin_safe.py)、[一般目标参数](../../../IPCCH/configs/forecasting_hyperparameters.json)、[q3参数](../../../IPCCH/configs/forecasting_hyperparameters_p3.json)。
- 特征数量已核对两边保存的矩阵/运行元数据；未做统一队列的控制变量重跑。本note不改变任何实验或结果。
