# D29 / A4：冻结候选后的确认诊断（范围及开发执行已授权）

### 1. 目的与预算

用户采用“首轮独立确认只做诊断”。不把C转成启用门槛；检验S上搜索增益在未参与本次搜索的同历史C上保留多少，再观察原目标月E3。保留D28 root单次增量机制。

新增**六根/六候选**：T={2018-02,2020-10} × H={4,8,12}，原r80/split seed42，L1/gt0，root来源，G锁H4=G1/H8=G4/H12=G2。主对照为D28的六个已完成rootinc候选，生产98adf48、运行`geoxgb-d28-rootinc-20261001`；D26/D27只作历史背景。不新增G拟合，不展开其他比例/seed/门槛/L或完整648。

### 2. 预定行池与确定性分割

先逐键恢复与D28完全相同的原r80 fitting/validation角色，保持fitting不变；原validation按以下无标签规则分成S/C，确认split seed固定42且单独记录。不得按C标签或结果平衡类别、补日期、换seed或重抽。

1. 每root新建Python3.12 `random.Random(42)`；area按数值升序，area内真实target_month升序。拒绝重复键。
2. 取validation行数为奇数的area列表（数值升序），用该rng shuffle。前`floor(奇数area数/2)`个area的S多一行，其余area的C多一行。
3. 按所有area升序，用同一rng各自shuffle其月份索引。前`floor(n/2)+该area的S额外行数`条为S，其余C。singleton只去一半，不移动回fitting。无validation的area不产生S/C行。

S∩C为空，S∪C恰为原validation，全局人数差≤1。这里是一半原validation，不声称整个历史池精确80/10/10；原r80本就有逐area ceil取整。保留完整几何及原fitting区域，不因C缺失/单例改变训练或目标cohort。

### 3. 搜索与冻结后评价

E1/q和E2只看S；E2仍比较当前父模型的四种路由，完整父S键精确crisis gain>0、父赢平局。局部训练仅用原fitting，root来源及一次L1=20、路径搜索额度≤80、最多五个二分层级不变。S应用原100行/20区/3日期支持底线，fitting原500行/50区/6日期/2观察类底线不变；不足继承父路由，不借C补支持。

搜索结束先冻结整份候选map、模型和路由，再以原predict-only路径评价全部C及原E3目标T。C不参与搜索的任何监督输入/计数/支持、候选停止/重试/模型选择；不要以C的类别计数参与root或child fitting资格。C评分不回写模型/地图，不合并S/C重训。C的无支持区域按既有路由fallback，预测行不得丢弃。

四类概率、argmax后crisis折叠、162特征、59个月窗、完整数据/地理、环境保持。C无门槛、不删候选、不剪枝、不触发root回退。E4只使用原E3规则，C正负分数均保留，不改Stage2算法。

### 4. 身份、对照及报告

新模式必须有区别于D28的root/candidate/run身份，记录原split及confirmation split的角色键；不覆盖旧输出。复用现有prepare/driver/predict/comparison及冻结输出身份，不新增通用实验/完整性框架。最终CLI在实施后记录。

新root按原fitting行拟合；解释结果前须验证新旧fitting键、原validation=S∪C、target键/truth、root SHA及root预测一致。原始G筛选预测来源固定`geoxgb-v1-20261001`，复用既有身份核查，不重复72fits。

逐H/T保存并重算S/C/E3的root/local四类和二元confusion、crisis F1及local−root；保留逐键真值/概率/硬预测/路由。D28旧validation预测可按同S/C键重算，但两部分均标明已被旧搜索使用；不把旧C称为独立确认。新S与旧完整validation不是同样本比较。报告每层rows/areas/dates/class与crisis支持、fallback、终端/不同booster、实际增量/搜索额度、覆盖可比分区重复和E4正权重。

ΔS−ΔC及ΔC−ΔE3仅描述数值差，不作因果分解：搜索样本减半也改变边界与支持，且存在时空相关、既有开发G选择及先前实验暴露。C只相对本次新搜索隔离；E3仍为开发评价，不作最终科学胜出判断。地图效用仍需之后合法Stage3开发比较，本轮不运行。

### 5. 正常/边界/失败及最小检查

正常：一个area原validation三行，按预定余数规则S/C分1/2或2/1；其fitting及root不变。边界：S不足支持时保留父预测，不能用C补齐；C极少正例仍报告，不设20正例门槛。错误：先用全validation搜索，再把半数重命名C；看到C负增益后重搜/删图/换seed；C评分后重训并仍宣称评价同一模型。

最小检查包括：奇数/singleton/空validation的确定性无标签分割；行键并集/互斥及原fitting不变；生产搜索路径中C标签置换不改变map/checkpoints/routes；C完整fallback预测；六候选身份及D28同root配对。测试通过不等于隔离保证已覆盖真实运行，需实际角色与产物核验。代码/环境/键/身份不匹配则报告incomplete并停止依赖解释，不自动替代日程或缩小cohort。

### 6. 执行顺序与停止

用户采用诊断范围，并指示“没问题stage3以外的数据怎么搞都行”，授权开发数据安排及当前六根实施/运行，不再逐项询问拆分/复用。先提交规划，原绑定Claude经native trellis-implement→native trellis-check→提交producer→fresh prepare/G复用→六候选→同键诊断/文档记录；controller/run/base保持，不重注册。完成后审阅证据。无论C/E3正负，均不自动启动完整648、Stage2/3、最终评价或audit close；不因无分裂或测试通过称过拟合解决。
