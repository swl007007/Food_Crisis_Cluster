# 从 Workflow Pack 1.0 到 GeoDT 可视化：把 HIGH/CRITICAL Review Loop 变成工程护栏

## 1. 开场：这不是一套流程，而是一套防止幻觉实现的刹车系统

### 1.1 为什么稳定版 workflow-pack 值得单独总结
Workflow Pack 1.0 的价值不在于多跑几个 slash command，而在于把“先理解、再承诺、后实现”的顺序固化成可重复的工程路径。

### 1.2 GeoDT 可视化为什么是一个合适的案例
GeoDT branch-tree interpretability 看似只是生成一张对比图，实际同时牵涉 legacy artifact、模型 checkpoint、可解释性语义、输出覆盖风险和审计复现。

### 1.3 HIGH/CRITICAL loop 的真正含义
每一次 HIGH 或 CRITICAL 问题都不应被看作 review 噪音，而应被看作流程中某个门禁失效后暴露出来的系统信号。

## 2. Workflow Pack 1.0 的稳定内核：从证据到实现的单向门禁

### 2.1 Setup：先规定 source-of-truth 顺序
稳定版 workflow-pack 首先明确 active feature 的 source-of-truth 顺序，避免 implementation-micro-plan、历史记忆或 migrated artifacts 反向覆盖当前 Speckit artifacts。

### 2.2 Brownfield Mode：已有系统默认受保护
Brownfield Mode 的核心假设是 legacy repository 中的既有行为默认受保护，任何行为变化都必须由当前 active feature 明确授权。

### 2.3 Evidence Pack：把“我以为”换成“我查到”
Evidence Pack 的作用是把候选路径、接口、测试、运行风险和 artifact 风险提前暴露，防止实现阶段再临时发现边界。

### 2.4 Specify / Clarify / Plan / Tasks / Analyze：让需求先经受交叉验证
Speckit artifacts 的主线不是文档仪式，而是让需求、计划、任务依赖和验收条件在动代码之前互相校验。

### 2.5 Micro-plan：把高层任务降解为可执行波次
Superpowers implementation micro-plan 的责任是把 spec、plan、tasks 和 evidence 翻译成执行顺序，同时不得擅自改变 scope 或 architecture。

### 2.6 Implement Constraint Prompt：把 TDD 和 Brownfield 护栏带进执行阶段
Implementation Constraint Prompt 将 TDD、validation、scope control 和 no-opportunistic-refactor 规则放在实现入口，避免 coding agent 在局部最优中越界。

### 2.7 Verify / Delta / Memory：实现完成后还要对账
实现后的 verify-run、implementation delta 和 memory compaction 共同回答三个问题：做成了吗、偏离了吗、哪些经验值得留下。

## 3. GeoDT 可视化案例：为什么“画一棵树”并不简单

### 3.1 最初的风险：把 root/global dt_rules 误当 branch-specific 解释
GeoDT 可视化的关键语义风险在于 root/global `dt_rules_*.csv` 并不能代表 branch-specific local DecisionTree 的解释。

### 3.2 正确的数据源：branch-specific local DecisionTree checkpoints
最终方案必须从分支级 checkpoint 中读取 local DecisionTree，并在图和 metadata 中说明没有使用 root/global rule export。

### 3.3 Archive discovery 不能无界搜索
在 legacy artifact 很多的项目里，archive discovery 必须是 bounded、可解释、可复现的，否则一次可视化会退化成不可审计的文件系统猜测。

### 3.4 Same-run provider 是例外，不是捷径
当目标 archive 缺少必要 artifacts 时，same-run provider 只能在兼容性证据被记录后作为受控补充，而不能绕过完整性检查。

### 3.5 Feature names 是 contract，不是装饰
Feature-name source 和 feature-count compatibility 必须被验证，因为一张漂亮的树图如果特征列错位，解释性就会变成误导。

## 4. HIGH/CRITICAL Loop 之一：Source-of-truth drift

### 4.1 症状：实现开始依赖错误的 artifact 层级
当 `.specify/feature.json`、migrated artifacts、micro-plan 或 tasks.md 之间出现 pointer drift 时，后续每一步都可能在错误 feature 上严谨地执行。

### 4.2 根因：reference-only 和 active-source 没有被硬隔离
Brownfield migrated artifacts 如果没有被明确标记为 reference-only，就很容易被误读为可授权行为变化的 active scope。

### 4.3 修复模式：显式 Active Feature Declaration
在 implementation-micro-plan 前声明 `active_feature: true`、implementation scope source 和 reference-only rule，可以把当前可执行范围固定下来。

### 4.4 经验：HIGH/CRITICAL 出现时先查 source-of-truth，而不是先改代码
一旦 review 指向 scope、pointer 或 artifact 冲突，正确动作是暂停实现并修复上游对齐，而不是在下游补丁中消化矛盾。

## 5. HIGH/CRITICAL Loop 之二：Implementation readiness 被过早放行

### 5.1 症状：行为改变任务早于 characterization 和 RED gate
GeoDT 开发中的一个关键教训是，能开始 T001 并不意味着可以开始所有 behavior-changing tasks。

### 5.2 根因：计划中没有区分“可探索”和“可修改”
如果 micro-plan 没有把 characterization、fixture、RED test 和实现任务分层，执行者会把“任务列表存在”误解为“全部任务可执行”。

### 5.3 修复模式：Implementation Start Gate
稳定做法是明确早期 characterization 和 RED-gate work 可立即开始，而 T024 之后的行为变化必须等待前置证据完成。

### 5.4 经验：HIGH/CRITICAL review 要逼迫流程回到 readiness gate
当 review 指出 readiness 问题时，不应只补一个测试，而应重写任务依赖和开始条件，让同类错误不能再次发生。

## 6. HIGH/CRITICAL Loop 之三：选择逻辑重复实现

### 6.1 症状：figure-generation 和 audit-only 各自实现 scoring
如果图像生成和 audit-only 模式分别实现 branch-pair scoring、readability gating 和 tie-break，两个模式迟早会产生不可解释的分歧。

### 6.2 根因：共享选择管线被排在了错误的用户故事之后
GeoDT micro-plan 的一次关键修正，是把 signature extraction、Jaccard scoring、tie-break 和 readability gating 提前为 US1 的基础路径。

### 6.3 修复模式：selection pipeline 只实现一次
正确结构是让 figure-generation 和 audit-only 都调用同一个 selection pipeline，并在 metadata 中证明两者选择结果一致。

### 6.4 经验：HIGH/CRITICAL 不只是在找 bug，也是在找重复事实来源
凡是 review 指出两条路径可能得出不同答案，都应优先合并决策源，而不是在两个分支上分别加保护。

## 7. HIGH/CRITICAL Loop 之四：输出 artifact 不可审计或可能被覆盖

### 7.1 症状：PNG 生成了，但无法证明为什么生成这张图
没有 metadata 的可视化只是图片文件，而不是可复现的诊断产物。

### 7.2 根因：artifact contract 没有覆盖 provenance、selection 和 safety
GeoDT 的最终 metadata 需要记录 selected archive、checkpoint paths、branch ids、pair score、readability、tie-break、rejected alternatives 和 output paths。

### 7.3 修复模式：metadata sidecar + no-overwrite preflight
图像生成前先检查目标 PNG 和 metadata 是否已存在，默认拒绝覆盖，并用 structured failure summary 报告冲突。

### 7.4 经验：HIGH/CRITICAL loop 结束条件必须包括 artifact validation
只有当 expected artifacts、schemas、paths、metadata 和 absence of unexpected outputs 都被验证后，输出类任务才算真正完成。

## 8. HIGH/CRITICAL Loop 之五：验证只证明代码能跑，不证明功能可信

### 8.1 症状：测试通过但没有覆盖真实风险
对于解释性可视化，单纯的 import smoke 或 happy-path test 不能证明 archive selection、feature compatibility 和 no-overwrite 行为可靠。

### 8.2 根因：validation 没有映射到风险清单
验证命令必须对应 Evidence Pack 和 Brownfield constraints 中列出的风险，而不是只运行最容易通过的测试。

### 8.3 修复模式：focused test + synthetic CLI smoke + artifact inspection
GeoDT 最终通过 focused diagnostic tests 和 synthetic end-to-end CLI smoke 同时覆盖内部逻辑和实际命令输出。

### 8.4 经验：HIGH/CRITICAL 的 loop 不以“改完”为终点，而以“复现风险被关闭”为终点
每个高严重度问题都应该留下一个可以再次运行的验证路径，否则下次 review 还会重新发现同一个类别的问题。

## 9. Workflow Pack 1.0 的工程原则总结

### 9.1 原则一：reference-only artifacts 只能约束，不能授权
M0 baseline、migrated artifacts 和 memory summaries 可以帮助识别 regression risk，但不能创建新的 implementation scope。

### 9.2 原则二：Brownfield 中的默认动作是最小变更
任何 opportunistic refactor、dependency upgrade、directory reshuffle 或 unrelated cleanup 都会扩大 review 面并稀释风险判断。

### 9.3 原则三：TDD 是行为变化的进入条件
如果任务会改变 runtime behavior，就必须先写或更新失败测试，除非用户明确批准并记录 TDD exception。

### 9.4 原则四：metadata 是 ML/visualization artifact 的一等输出
在模型诊断和可解释性任务中，metadata 不只是辅助文件，而是让输出可信、可审计、可复现的核心 contract。

### 9.5 原则五：review loop 的目标是降低下一轮 review 的熵
好的修复不是让当前 reviewer 闭嘴，而是让下一轮 HIGH/CRITICAL 问题在结构上更难出现。

## 10. 可复用模板：下一次 Brownfield 可视化功能应该怎么走

### 10.1 第一步：声明 active feature 和 reference-only 边界
任何新功能开始前，都应先确认当前 active feature artifact set 是唯一能授权实现的来源。

### 10.2 第二步：列出 artifact risk 和 overwrite risk
凡是会读取历史产物或写入新图表的任务，都应在 Evidence Pack 中列出输入规模、路径敏感性、覆盖风险和安全 smoke command。

### 10.3 第三步：把 selection、rendering、audit 拆成可验证 contract
可视化任务应明确区分“选择什么”“如何渲染”“如何审计”，并尽量让选择逻辑成为共享底座。

### 10.4 第四步：让 HIGH/CRITICAL loop 更新流程，而不只是更新代码
当严重问题出现时，修复应反映到 source-of-truth、task dependency、micro-plan gate 或 validation contract 中。

### 10.5 第五步：用 Implementation Delta 对账
实现结束时必须说明 allowed change surface 是否被遵守、forbidden paths 是否被触碰、是否有 accepted deviation，以及哪些失败是 pre-existing。

## 11. 结尾：稳定流程不是降低速度，而是减少返工

### 11.1 从 GeoDT 案例回看 workflow-pack 的收益
GeoDT 可视化之所以最终可交付，不是因为一次性写对了代码，而是因为每轮 HIGH/CRITICAL 都被转化成更强的证据、门禁和验证。

### 11.2 对未来工作的建议
未来在 legacy ML 项目中开发解释性、预测或可视化功能时，应默认采用 Workflow Pack 1.0 的 Brownfield-SDD-TDD 路径，并把 review loop 当作流程改进信号。
