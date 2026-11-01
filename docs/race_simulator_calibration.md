# 模拟器历史校准记录（2026-11-01）

已完成一次实际系数拟合、开发集五折选择和独立日期留出验证。候选参数暂未替换页面默认参数：用时更接近实际，但保持原随机波动强度时，排名覆盖反而下降。未修改生产 Agent 打分或原始预测、赛果、审计。

## 数据与划分

只使用最新 Render Agent LLM 审计和赛果，不使用 v1—v6 数据。下载时线上新增赛果，已对 30,806 个文件核对 SHA-256；变化的四个文件保存在独立补丁目录。原始快照没有被覆盖。

- 比赛日期：2026-07-07 至 2026-10-31，共 5,615 场符合平地模型条件且有前三名赛果的 full 样本。
- 开发集：4,399 场，中央 834、地方 3,565；按日期分五折，每场使用折外参数评估。
- 独立留出：2026-10-08 至 2026-10-31，共 1,216 场，中央 230、地方 986。这些比赛不参与系数拟合和混合比例选择。
- 下载审计有 5,737 场 full，prediction_only 为零；另有 1,692 场已结算预测缺少审计，不能重算。排除非平地 36 场、前三赛果不完整 8 场、当天尚未完整的比赛日 60 场、缺赛果 18 场。
- 有效 full 中 250 场没有近走；共 2,966 匹没有近走，保留明示中间值。它们可以约束总体用时，不能提供个体能力证据。

模拟输入只保留严格早于比赛日的原始近走，脚质从历史通角位置重建，不采用保存的 Agent 分数或分析。赔率、人气和目标赛果均不进入输入。赛前存档的准确发走时间不可得，因此这是按日期重建的历史验证，不能等同于已冻结输入的前瞻验证。

最后 600 米的实测标签通过后续 Agent 审计的近走找回，要求比赛、马匹身份、日期、距离、头数、名次一致；只加入独立标签表，不进入该场赛前资料。留出集只有 2,970 匹次有末段标签，存在后续参赛选择偏差。

## 结果

以下关闭随机扰动，用前端原版引擎计算。完赛时间误差有 12,130 匹次标签。

|独立留出指标|手工参数|校准候选|
|---|---:|---:|
|逐马完赛时间平均绝对误差|3.48 秒|1.93 秒|
|最后 600 米平均绝对误差|3.64 秒|1.14 秒|
|模拟前五覆盖实际前三席位|2,366 / 3,648|2,453 / 3,648|
|覆盖率|64.86%|67.24%|
|前三全部覆盖场次|306|347|
|至少覆盖两匹场次|885|933|
|冠军进入模拟前五|72.53%|73.52%|
|逐马名次平均绝对误差|3.02 名|2.89 名|

按日期区组自助抽样 2,000 次，总体覆盖率变化的 95% 区间约为 +1.55～+3.26 个百分点。改进主要来自地方；中央冠军进入模拟前五由 62.61% 降至 60.87%，仅 230 场，不能宣称中央稳定增强。

保留页面的原随机波动强度 1，使用预先固定的五个种子后，平均覆盖率由 **60.12% 降至 57.32%**；完赛时间误差仍由 3.53 秒降至 2.29 秒。说明目前只把用时校准得更接近，还没有把完整随机模拟校准好。不能把关闭随机时的改进直接当作动画或真实胜率改善。

候选保持冻结，不根据这批留出比赛继续调参。默认参数保留原值。后续若校准随机状态，需要新的独立验证，不能反复使用本次留出集选择参数。

## 系数与仍然存在的假设

有界稳健最小二乘同时约束完赛用时、末段用时、实际前三与其他马的相对排序，并向原参数收缩。开发集比较原值、50% 混合和完整候选，按前五覆盖实际前三的主要指标选择完整候选；最终启用判断还检查完整随机模拟。

主要变化：草地基准速度 16.8→17.7108，泥地 15.8→16.6216，新增地方速度修正 -0.7408、距离对数项 -1.6301；末段能力速度系数 1.9→0.9495，能量消耗 0.67→1.2654，持久力节能 0.3→0.0053。基准速度是公式偏置，不等于真实平均速度，各项必须整体使用。

早段能力和持久力节能系数接近下界，只表明当前代理指标在这套模型里缺少额外信息。体力消耗、疲劳、冲刺之间可以相互补偿，没有生理体力实测值，不能单独解释成真实生理系数。坡度响应、加速度、疲劳阈值、进路、外绕成本、随机状态分布仍为预设；没有验证真实途中轨迹，也没有模拟碰撞和堵路。

## 文件与复现

完整报告、图表、参数、协议与逐场结果位于 `analysis_outputs/simulator_calibration_20261101/`：

- `report.md`、`holdout_comparison.png`：完整结果与图。
- `frozen_model.json`：在留出评估前保存的系数、五折系数和开发集选择结果。
- `protocol.json`、`fit_protocol.json`：数据来源、校验和、分折、目标、边界、固定种子和依赖版本。
- `summary.json`、`race_metrics.json`、`validation.json`：原版引擎结果、逐场指标和独立重算。
- `application_decision.json`：候选未启用及原因。
- `run_source/`：本次执行时的五份原始源码，校验和与拟合协议一致。

准备脚本拒绝覆盖已有分析目录，拟合脚本拒绝覆盖已冻结模型。依赖只安装在独立分析目录，不加入生产依赖。复现时使用新的输出目录：

```powershell
python -m pip install --target analysis_outputs/simulator_calibration_runtime numba
python scripts/prepare_simulator_calibration.py --snapshot analysis_outputs/render_agent_snapshot_20261101_simulator_calibration --overlay analysis_outputs/simulator_snapshot_reconciliation_20261101 --before-date 2026-11-01 --output analysis_outputs/simulator_calibration_new_run
node scripts/simulator_calibration_bridge.mjs prepare analysis_outputs/simulator_calibration_new_run/cases.json analysis_outputs/simulator_calibration_new_run/prepared.json
python scripts/calibrate_simulator.py fit --directory analysis_outputs/simulator_calibration_new_run
python scripts/calibrate_simulator.py evaluate --directory analysis_outputs/simulator_calibration_new_run
python scripts/report_simulator_calibration.py analysis_outputs/simulator_calibration_new_run
```

本次 NumPy 2.4.6、Numba 0.68.1，SciPy 版本见拟合协议。若要逐字节核对执行源码，使用 `run_source/` 和 `fit_protocol.json`；当前脚本额外增加了拒绝覆盖冻结模型的保护。所有最终指标来自 JavaScript 原版，加速版只用于拟合；两者用时差平均 0.00168 秒、最大 0.12728 秒。

验证包括 17 项前端模拟测试、2 项数据隔离测试、前端构建，以及用独立代码重新核对覆盖席位和两种用时误差。图表已检查。
