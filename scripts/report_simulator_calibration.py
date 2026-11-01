"""Independently reconcile key held-out metrics and render an inspectable report."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1] / 'analysis_outputs/simulator_plot_cache'))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def load(directory, name):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def report(directory):
    protocol = load(directory, "protocol.json"); summary = load(directory, "summary.json")
    model = load(directory, "frozen_model.json"); prepared = load(directory, "prepared.json")
    cases = prepared["cases"]; hold = [c for c in cases if c["split"] == "holdout"]
    # Independent set/absolute-error arithmetic, without the fitter's helpers.
    baseline = {r["race_id"]: r["predictions"] for r in load(directory, "exact_baseline.json")}
    candidate = {r["race_id"]: r["predictions"] for r in load(directory, "exact_fitted_candidate.json")} if (directory / "exact_fitted_candidate.json").exists() else {r["race_id"]: r["predictions"] for r in load(directory, "exact_calibrated.json")}
    independent = {}
    for name, predictions in (("baseline", baseline), ("fitted_candidate", candidate)):
        covered = 0; clock = []; closing = []
        for c in hold:
            labels = {r["horse_no"]: r for r in c["labels"]}; predicted = sorted(predictions[c["race"]["race_info"]["race_id"]], key=lambda r: (r["time"], r["horse_no"]))
            covered += sum(labels.get(r["horse_no"], {}).get("finish", 999) <= 3 for r in predicted[:5])
            for r in predicted:
                lab = labels.get(r["horse_no"], {})
                if lab.get("time") is not None: clock.append(abs(r["time"] - lab["time"]))
                if lab.get("final_3f") is not None: closing.append(abs(r["final_3f"] - lab["final_3f"]))
        expected = summary["holdout"][name]
        assert covered == expected["covered_seats"]
        assert abs(np.mean(clock) - expected["time_mae"]) < 1e-10
        if closing: assert abs(np.mean(closing) - expected["closing_mae"]) < 1e-10
        independent[name] = {"covered_seats": covered, "time_mae": float(np.mean(clock)), "closing_mae": float(np.mean(closing)) if closing else None}
    # Date bootstrap for the predetermined fitted candidate, including when OOF
    # chose the original baseline. No decisions are made from these intervals.
    day_values = collections.defaultdict(lambda: np.zeros(7))
    for c in hold:
        rid = c["race"]["race_info"]["race_id"]; labels = {r["horse_no"]: r for r in c["labels"]}; values = []
        for pred in (baseline[rid], candidate[rid]):
            pp = sorted(pred, key=lambda p: (p["time"], p["horse_no"]))
            seats = sum(labels.get(r["horse_no"], {}).get("finish", 999) <= 3 for r in pp[:5])
            times = [abs(p["time"] - labels[p["horse_no"]]["time"]) for p in pp if labels.get(p["horse_no"], {}).get("time") is not None]
            closes = [abs(p["final_3f"] - labels[p["horse_no"]]["final_3f"]) for p in pp if labels.get(p["horse_no"], {}).get("final_3f") is not None]
            values.append((seats, sum(times), sum(closes), len(times), len(closes)))
        a,b = values
        day_values[c["race"]["race_info"]["race_date"]] += [b[0]-a[0], 3, b[1]-a[1], a[3], b[2]-a[2], a[4], 1]
    day_array = np.array(list(day_values.values())); rng = np.random.default_rng(20261101)
    boot = np.array([day_array[rng.integers(0,len(day_array),len(day_array))].sum(axis=0) for _ in range(2000)])
    intervals = {name: np.quantile(boot[:,a]/boot[:,b], [.025,.975]).tolist() for name,a,b in (("recall_delta",0,1),("time_mae_delta",2,3),("closing_mae_delta",4,5))}
    validation = {"independent_reconciliation": independent, "fitted_candidate_date_bootstrap_95_ci": intervals, "protocol_sha256": hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest(), "fit_protocol_sha256": hashlib.sha256((directory / "fit_protocol.json").read_bytes()).hexdigest(), "frozen_model_sha256": hashlib.sha256((directory / "frozen_model.json").read_bytes()).hexdigest()}
    stochastic = {label: {metric: float(np.mean([r[metric] for r in summary['stochastic_holdout'][label]])) for metric in ('top3_recall','time_mae','closing_mae')} for label in ('baseline','fitted_candidate')}
    validation['stochastic_seed_mean'] = stochastic
    decision = {'default_enabled': False, 'selected_development_blend': model['chosen_blend'], 'reason': 'Mixed validation: clocks improve, but Top5 coverage regresses with unchanged default random variation; central winner Top5 also decreases. Preserve candidate; retain default coefficients.', 'parameters_changed_after_holdout': False, 'production_agent_changed': False}
    (directory / 'application_decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    (directory / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    a,b = (summary["holdout"][k] for k in ("baseline", "fitted_candidate"))
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1,4,figsize=(15,4.8), layout="constrained")
    metrics = [("完赛时间平均绝对误差", "time_mae", "秒", None), ("最后600米平均绝对误差", "closing_mae", "秒", None), ("模拟前五覆盖实际前三", "top3_recall", "%", (0,100))]
    for ax,(title,key,unit,limit) in zip(axes[:3],metrics):
        vals = [a[key], b[key]]
        if key == "top3_recall": vals = [v*100 for v in vals]
        ax.bar([0,1],vals,color=["#8896a8","#118977"],width=.55)
        ax.set_xticks([0,1],["原手工参数","拟合候选参数"]);ax.set_title(title);ax.set_ylabel(unit)
        ax.spines[["right","top"]].set_visible(False);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
        ax.set_ylim(limit or (0,max(vals)*1.27))
        for i,v in enumerate(vals):ax.text(i,v+ax.get_ylim()[1]*.025,f"{v:.2f}{unit}",ha="center",fontweight="bold")
    ax=axes[3]; vals=[stochastic[k]['top3_recall']*100 for k in ('baseline','fitted_candidate')]
    ax.bar([0,1],vals,color=['#8896a8','#118977'],width=.55);ax.set_xticks([0,1],['原手工参数','拟合候选参数']);ax.set_ylim(0,100);ax.set_title('开启随机波动：前三覆盖率');ax.set_ylabel('%');ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    for i,v in enumerate(vals):ax.text(i,v+2.5,f'{v:.2f}%',ha='center',fontweight='bold')
    first,last = min(c["race"]["race_info"]["race_date"] for c in hold),max(c["race"]["race_info"]["race_date"] for c in hold)
    fig.suptitle(f"独立日期留出：{first}—{last}，{len(hold):,}场｜候选暂未启用\n前三图关闭随机扰动；末图为5个固定种子的均值。末段仅{a['closing_n']:,}匹次有标签",fontsize=14)
    fig.savefig(directory / "holdout_comparison.png", dpi=170);plt.close(fig)
    counts = collections.Counter((c["split"],c["scope"]) for c in cases)
    lines = ["# 赛马模拟器历史校准与独立验证", "", "本报告比较当前手工参数与仅用开发集拟合的候选参数。最终指标均由前端实际模拟引擎重新计算，不使用目标比赛答案控制过程。", "", "## 数据与隔离", "", f"- 来源：`{protocol['snapshot']}`，仅 Agent LLM 审计、赛前原始输入和赛果。", f"- 可重算且满足平地模型与赛果条件：{len(cases):,}场；开发集{protocol['split_counts']['development']:,}场，按日期五折；独立留出{len(hold):,}场、{len(day_array)}个日期（{first}—{last}）。", f"- 开发集中央{counts[('development','central')]:,}、地方{counts[('development','local')]:,}；留出集中央{counts[('holdout','central')]:,}、地方{counts[('holdout','local')]:,}。", "- 近走日期必须严格早于比赛日，同时排除目标比赛本身、无日期记录。脚质从历史通角位置重建，不采用已保存的 Agent 分析或分数。赔率、人气、目标赛果均不进入模拟输入。", "- 最后600米标签来自后续 Agent 审计中的近走，按比赛ID、马ID（缺失时马名）、日期、距离、头数、名次一致后匹配；冲突保留缺失。它们只进入标签表，绝不补进目标赛前输入。", "- 原始资料按历史日期重建；未取得每场准确发走时间，不能证明所有存档均在发走前冻结。这是历史重建验证，并非前瞻实盘验证。", "", "## 独立留出结果", "", "下表‘候选’是预先拟合并冻结的完整候选；是否采用由开发集折外指标决定，留出结果不会反过来选系数。时间误差按有标签的马计算，排名指标按场计算。", "", "|指标|原手工参数|拟合候选|", "|---|---:|---:|"]
    central_a=summary['holdout_by_scope']['baseline']['central'];central_b=summary['holdout_by_scope']['fitted_candidate']['central']
    lines[2:2] = ["**结论：可附限制分享，暂不替换默认参数。** 关闭随机扰动时，用时与排名都有改进；保留原随机波动强度后，平均前三覆盖率却由"+f"{stochastic['baseline']['top3_recall']:.2%}降至{stochastic['fitted_candidate']['top3_recall']:.2%}。中央冠军进入模拟前五的比例也由{central_a['winner_top5']:.2%}降到{central_b['winner_top5']:.2%}。候选参数保持冻结，不根据留出集继续调参；页面默认模型和生产 Agent 打分保持原样。", ""]
    if protocol['verified_overlay']:
        verification=load(Path(protocol['verified_overlay']),'verification.json')
        lines.insert(lines.index('## 数据与隔离')+2, f"快照已核对{verification['verified_files']:,}个文件的SHA-256；下载期间变化的文件使用独立补丁 `{protocol['verified_overlay']}`，原快照保持不变。\n")
    lines=[line.replace('是否采用由开发集折外指标决定','候选混合比例由开发集折外指标决定，启用还需检查完整随机模拟').replace(f"{intervals['recall_delta'][0]:+.2%}～{intervals['recall_delta'][1]:+.2%}",f"{intervals['recall_delta'][0]*100:+.2f}～{intervals['recall_delta'][1]*100:+.2f}个百分点") for line in lines]
    for label,key,fmt in (("实际前三席位被模拟前五覆盖", "covered_seats", ",.0f"),("覆盖率", "top3_recall", ".2%"),("前三全部覆盖场次", "all3_races", ",.0f"),("至少两匹覆盖场次", "at_least2_races", ",.0f"),("冠军进入模拟前三", "winner_top3", ".2%"),("冠军进入模拟前五", "winner_top5", ".2%"),("冠军平均模拟名次", "winner_mean_rank", ".2f"),("冠军完全命中率", "winner_accuracy", ".2%"),("逐马完赛时间平均绝对误差（秒）", "time_mae", ".3f"),("最后600米平均绝对误差（秒）", "closing_mae", ".3f"),("逐马着顺平均绝对误差（名）", "rank_mae", ".3f")):
        lines.append(f"|{label}|{a[key]:{fmt}}|{b[key]:{fmt}}|")
    lines += ["", f"完赛时间标签{a['time_n']:,}匹次；末段标签{a['closing_n']:,}匹次。末段标签依赖马匹后续参赛，存在覆盖与选择偏差，不能推广为全部马匹精度。", "", "![独立留出比较](holdout_comparison.png)", "", "候选减去基线的日期区组自助抽样95%区间（2,000次）：", "", f"- 前三覆盖率变化：{intervals['recall_delta'][0]*100:+.2f}～{intervals['recall_delta'][1]*100:+.2f}个百分点。", f"- 完赛时间误差变化：{intervals['time_mae_delta'][0]:+.3f}～{intervals['time_mae_delta'][1]:+.3f}秒。", f"- 末段误差变化：{intervals['closing_mae_delta'][0]:+.3f}～{intervals['closing_mae_delta'][1]:+.3f}秒。", "", "## 中央与地方", "", "|范围|场次|原覆盖率|候选覆盖率|原时间误差|候选时间误差|", "|---|---:|---:|---:|---:|---:|"]
    for scope,label in (("central","中央"),("local","地方")):
        aa=summary['holdout_by_scope']['baseline'][scope];bb=summary['holdout_by_scope']['fitted_candidate'][scope]
        if aa['races']: lines.append(f"|{label}|{aa['races']}|{aa['top3_recall']:.2%}|{bb['top3_recall']:.2%}|{aa['time_mae']:.3f}秒|{bb['time_mae']:.3f}秒|")
    lines += ["", "## 开发集选择与系数", "", "开发集按日期分五折，同一日期所有比赛放在同一折，每场使用其所在折之外拟合的系数。它混合较早与较晚开发日期，属于模型选择验证；真正时间外推证据为上述更晚留出集。", "", "|候选混合比例|折外前三覆盖率|折外完赛误差|折外末段误差|", "|---|---:|---:|---:|"]
    for blend,m in model['development_oof'].items(): lines.append(f"|{float(blend):.0%}|{m['top3_recall']:.2%}|{m['time_mae']:.3f}秒|{m['closing_mae']:.3f}秒|")
    lines += ["", f"按开发集主要指标选出的候选混合比例：**{model['chosen_blend']:.0%}**（0%表示保留原参数）。留出集结果不参与该决定。", "", "|参数|原值|开发集拟合值|", "|---|---:|---:|"]
    labels = ["草地基准速度（米/秒）","泥地基准速度（米/秒）","地方速度修正（米/秒）","距离对数速度系数","历史能力速度系数","早段位置速度系数","冲刺基础加速","末段能力速度系数","后方脚质冲刺增量","能量消耗系数","持久力节能系数","疲劳减速系数"]
    for label,key in zip(labels,model['parameters']): lines.append(f"|{label}|{prepared['parameters'][key]:.4f}|{model['fitted_parameters'][key]:.4f}|")
    lines += ["", "能量消耗、持久力节能和疲劳减速是联合正则化拟合的潜在状态系数，不能各自解释为真实生理测量。未校准的假设包括坡度速度/耗力响应、疲劳阈值、加速度、进路与外绕成本、随机状态分布、碰撞与堵路。", "", "## 单次随机模拟", "", "保持原随机分布和波动强度1，使用预先固定的5个种子。以下为5次结果范围，不是置信区间。", "", "|参数|前三覆盖率范围|完赛误差范围|末段误差范围|", "|---|---:|---:|---:|"]
    for name,label in (("baseline","原手工"),("fitted_candidate","拟合候选")):
        ms=summary['stochastic_holdout'][name]
        lines.append(f"|{label}|{min(m['top3_recall'] for m in ms):.2%}～{max(m['top3_recall'] for m in ms):.2%}|{min(m['time_mae'] for m in ms):.3f}～{max(m['time_mae'] for m in ms):.3f}秒|{min(m['closing_mae'] for m in ms):.3f}～{max(m['closing_mae'] for m in ms):.3f}秒|")
    no_history=sum('recent_runs' in m['missing_inputs'] for m in protocol['manifest'])
    lines += ["", "## 数据覆盖与可复核性", "", "```json", json.dumps(protocol['quality'], ensure_ascii=False, indent=2), "```", "", f"- 有效full样本中，{no_history}场缺少近走，{protocol['quality']['horses_no_history']:,}匹没有近走；仍保留它们，能力指标使用明示中间值。有效样本均有出马表和前三名赛果；原审计缺少result但能由同快照赛果补齐的，仅在派生标签表关联。", "- 不包含旧分数组件调权回测；这是 full 输入上的模拟器重算。prediction_only、缺少出马表/赛果、非平地和不支持距离等样本不用于拟合。未读取v1—v6预测器数据，也未修改生产 Agent 打分。", f"- 加速拟合版与最终原版引擎逐马用时差：平均{summary['compiled_vs_exact_time_difference']['mean_seconds']:.6f}秒，最大{summary['compiled_vs_exact_time_difference']['max_seconds']:.6f}秒。最终表格均使用原版。", "- `protocol.json`保存分折、来源和文件校验和；`fit_protocol.json`保存目标、边界、种子；`frozen_model.json`在留出评估前保存系数；`validation.json`保存独立重算与日期区组区间；`race_metrics.json`保存逐场指标。", "- 完整原始快照保持原样；所有拟合和报告均在独立分析目录。", "", "复现顺序：`prepare_simulator_calibration.py` → `simulator_calibration_bridge.mjs prepare` → `calibrate_simulator.py fit` → `calibrate_simulator.py evaluate` → `report_simulator_calibration.py`。", "", "方法参考：[SciPy有界稳健最小二乘](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)、[scikit-learn训练与测试隔离](https://scikit-learn.org/stable/modules/cross_validation.html)。"]
    (directory / "report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(validation,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path);args=parser.parse_args();report(args.directory)
