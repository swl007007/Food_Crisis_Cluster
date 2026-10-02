# D32 / A7：Stage1分配证据导出（工程修复，已完成）

状态：用户指示“先回到一阶段，解决root的工程问题以后再探索怎么解决一阶段的过拟合问题；二阶段公式问题先保留”。本方案**取代**此前D32四图Stage2提案（推迟，不实现）。只做Stage1导出语义修复：不改搜索、拟合、路由、预测或任何评分；不是过拟合修复。审计run ed632775/base/session不变；无模型实验批次、Stage2/3、最终评价或close。

## 1. 问题（只读追踪，见`research/zero-fit-dev-baseline-and-root-label-routing.md`）

无搜索行（S）的区从不进入scan（scan组只来自分支validation行，`transformation.py:293-319`），按设计留在父/root（`:733-742`，`parent_kept_fitting` `:763`，现有运行仅在root层非零）；其fitting行只训练root，预测走root（`native_xgb.py:305-325`）——路由一致，非bug。但`correspondence_table.csv`把`""`导出为`root`（`main_model_GF.py:121-128`），混合了：无搜索父回退、搜索后未分裂的真root、以及（未导出的）仅目标区root回退。下游不能把预测回退读作“与其他无搜索区共享已学习区域”。

## 2. 新导出 `assignment_evidence.csv`（每候选一份）

由现有`run_candidate`在候选冻结后（与`correspondence_table.csv`同处）写出；不重训、不改任何预测。

宇宙：搜索输入`gtrain`的区 ∪ 目标区`gtest` ∪（D29+模式）C区。现有D29–D31保存运行中C/unused区均有fitting行，C-only区为0；若出现则按下表计数，不报错掩盖。

| 列 | 定义 |
|---|---|
| `FEWSNET_admin_code` | 区键 |
| `prediction_branch_id` | 与现有路由相同：`get_X_branch_id_by_group(area, s_branch)`，`""`记`root`（=correspondence及测试路由） |
| `spatial_partition_id` | `search_rows ≥ 1`时等于`prediction_branch_id`；否则`s-1` |
| `search_rows` / `fitting_rows` | 搜索输入中该区`x_set==1` / `x_set==0`行数 |
| `target_rows` | 该区目标月行数 |
| `assignment_status` | `searched_assigned`（有搜索、非root终端）；`searched_root`（有搜索、终端为root；现逻辑下仅当root未接受分裂）；`unsearched_fit_fallback`（无搜索、有fitting）；`target_only_fallback`（无搜索/fitting、有目标行；相对搜索输入的仅目标，含同时有C行者）；`confirmation_only_fallback`（无搜索/fitting/目标、仅C行，预期0）。优先级：search→fitting→target→C |
| `confirmation_rows` | 该区C行数（只计数，不影响空间支持） |
| `routed_booster_is_root` | 描述字段：路由终端在现有saved_log末次记录的booster SHA是否等于root booster（root终端为真）。不作为掩码或支持判断；若无法可靠取得则不写此列，不另建注册表 |

规则：有搜索的未分裂root保留空间`root`；有搜索、命名分支但用root booster副本者保留其空间分支；无搜索/仅目标/仅C的root回退空间记`s-1`。空间支持只由实际搜索行数决定，不用评分、标签、C结果或booster哈希推断。

## 3. 契约与兼容

- `correspondence_table.csv`保持不变，定位为兼容的**预测路由**导出，不是已学习空间分配的证明。`candidate.json`增加两段：`routing_export`（文件、契约说明）与`assignment_evidence`（schema `d32-v1`、文件SHA-256、按状态的区数/目标行数汇总、契约“Stage1空间证据权威”）。
- 新文件进入完成链：`run_stage1`复制并记录哈希；仅当候选声明该schema时才要求该文件。D28–D31既有固定运行不改、不重跑，比较/接受路径继续可用。
- 不改旧产物，不让旧correspondence在新下游用途中静默成为权威。Stage2消费端对齐推迟；`DOWNSTREAM_ALIGNED=False`保持；不改Stage2源码/方法、不运行Stage2。
- 全部预测/评价键、C/E3概率、E1/E2/E3/E4、支持底线、root拟合与子拟合逻辑不变。

## 4. 对D30/D31解释的修正（不改数值）

D30的root/global拟合角色完全不变。分支专属拟合池的变化是搜索/分区成员改变的**诱发后果**：子局部拟合池按设计只含被分配到该分支的区，D30收窄S使2267–2457个区（D29为180–277）留在root，其fitting行因此不进入任何子拟合。这不是第二条独立改变的拟合规则，也不是违反既有实现契约或数据泄漏/拟合bug；只是“只改变搜索历史”的表述需补充这一诱发后果。D31固定逐区S计数，但各seed实际学到的分支及其拟合池仍可不同。E3局部−root差只来自有搜索区的目标行（D30/D31为2268–2462/5364行在root上，D29为182–282）。

## 5. 检查（最小，复用现有生产夹具）

真实`run_candidate`夹具覆盖：无搜索+有fitting/目标区；有搜索的root未分裂；有搜索的命名分支（含root booster副本时`routed_booster_is_root`为真，若该列写出）；仅目标区；C隔离（改变C标签不改变证据文件）。断言：证据行与宇宙1:1，计数一致，`spatial_partition_id=='s-1'`当且仅当`search_rows==0`；原路由、概率、评分与`correspondence_table.csv`字节不变；旧固定运行仍可接受。“不变”证据须分别标明：旧/新代码对同一夹具的实际重放对比、对未改动数值代码的审查、或仅新代码夹具断言——只报告实际验证了哪一种，不以一种冒充另一种。全部新预期区类型（含仅目标区）都要有测试。另以保存的D31成员与correspondence在原运行之外按键推导支持计数做核对（标明是推导，不称为新的producer执行）。不做强制旧运行重跑，不新建完整性框架。

## 6. 流程与停止

规划提交→原绑定Claude native trellis-implement→native trellis-check→测试→代码提交→独立证据核对/文档。无模型实验批次、Stage2/3、最终评价或close。完成后回到Stage1过拟合研究（不再在同六个目标月上反复微调），由监督方主导新规格。

## 7. 工程结果（2026-10-01）

- 代码c079b75（规划85f074c）；测试77 OK，exit 0。native check与精确再检均无契约性问题；证据性：命名root副本分支仅纯函数测试，`x_set`假定为0/1（唯一生产者写0/1），接受路径不交叉核对`candidate.json`与`completion.json`中的SHA。
- **实际旧/新重放**（同一合成生产夹具经真实`run_candidate`；旧=85f074c包代码经`git archive`导出，新=c079b75）：correspondence、target/heldout预测、X_branch_id、branch_table、s_branch逐字节一致；validation/e2/confirmation预测解压后一致（gzip头含时间戳）；全部checkpoint `.ubj/.json`逐字节一致；`candidate.json`与返回记录除新增`routing_export`/`assignment_evidence`、`timings`及临时checkpoint目录路径外完全一致；新代码只多出`assignment_evidence.csv`。这是该夹具上的实际重放，不是对所有数据的等价证明。
- **保存D31推导**（非新producer执行）：用已提交纯函数`assignment_evidence()`对保存的D31 membership键与各候选`s_branch.pkl`推导：18候选状态计数、目标行与区数与监督方独立stdlib推导（`/tmp/d32_independent_saved_support.json`）全部一致；路由列等于保存的correspondence及target路由；`s-1`当且仅当零搜索行；全部无搜索区路由root。含有搜索未分裂root（2846/2857/2977区）与晚期H8/H12的141个仅目标区。
- 数值过拟合未解决；Stage2消费端对齐推迟。
