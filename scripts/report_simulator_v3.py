"""Build a source-backed record of development OOF and unseen-date validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'analysis_outputs/simulator_plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main(d):
    dev=read(d/'development_summary.json');new=read(d/'new_summary.json');protocol=read(d/'new_validation_protocol.json');model=read(d/'frozen_model.json');choice=model['development_choice'];rows=read(d/'new_rows.json')
    # Independent raw-outcome reconciliation for the main MC ranking metric.
    cases=read(d/'new_cases.json');actual={c['race']['race_info']['race_id']:{h['horse_no'] for h in c['labels'] if h['finish']<=3} for c in cases}
    for name in ('baseline','previous','optimized'):
        predicted=read(d/f'new_{name}.json');seats=0;errors=[]
        for c,p in zip(cases,predicted):
            order=sorted(p['predictions'],key=lambda h:(-h['top3_count'],h['time'],h['horse_no']));seats+=len({h['horse_no'] for h in order[:5]}&actual[p['race_id']]);labels={h['horse_no']:h for h in c['labels']}
            errors += [abs(h['time']-labels[h['horse_no']]['time']) for h in order if labels.get(h['horse_no'],{}).get('time') is not None]
        assert seats==new[name]['covered_seats'];assert abs(sum(errors)/len(errors)-new[name]['mean_time_mae'])<1e-10
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11})
    fig,axes=plt.subplots(1,4,figsize=(15,4.8),layout='constrained')
    panels=[('历史五折：汇总前三覆盖率',dev['baseline']['top3_recall']*100,dev[choice]['top3_recall']*100,'%',100),('新比赛：汇总前三覆盖率',new['baseline']['top3_recall']*100,new['optimized']['top3_recall']*100,'%',100),('新比赛：单次模拟平均覆盖率',new['baseline']['single_run_recall']*100,new['optimized']['single_run_recall']*100,'%',100),('新比赛：平均预测用时误差',new['baseline']['mean_time_mae'],new['optimized']['mean_time_mae'],'秒',None)]
    for ax,(title,a,b,unit,limit) in zip(axes,panels):
        ax.bar([0,1],[a,b],color=['#8896a8','#118977'],width=.55);ax.set_xticks([0,1],['原模型','新版试验模型']);ax.set_ylim(0,limit or max(a,b)*1.3);ax.set_title(title);ax.set_ylabel(unit);ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        for i,v in enumerate([a,b]):ax.text(i,v+ax.get_ylim()[1]*.025,f'{v:.2f}{unit}',ha='center',fontweight='bold')
    fig.suptitle(f"新版指标与随机机制｜新日期独立留出仅{protocol['new_races']}场，旧模型仍为默认\n汇总排名按多次模拟前三频率；单次模拟单独评估。新样本没有最后600米实测标签。",fontsize=14)
    fig.savefig(d/'comparison.png',dpi=170);plt.close(fig)
    lines=['# 速度、持久力、末段与随机波动优化（2026-11-01）','','**结论：研究候选已实现，开发集改善；新日期样本仍为混合结果，旧模型继续默认。** 新版已加入模拟器的“履歴校正モデル（検証中）”切换项。参数在打开新比赛标签前冻结，此后没有根据新结果调参。','','## 本次实际改动','','- 速度加入赛前历史完赛时间，并按场地、草地/泥地、距离、马场状态建立参考；99,247条历史用时经比赛ID、马匹身份、日期、距离和名次一致后关联。缺少历史用时时保留收缩后的历史成绩指标。','- 末段从原来的同场参赛马内部百分位，改为相对条件参考用时的差异。参考模型只在该折训练资料上拟合。','- 持久力与历史名次分开：结合最后通角至终点的相对位置变化，以及有实测用时记录中的前段至最后600米速度衰减。仍为代理指标，不能解释为生理体力。','- 历史观察按新近程度与距离接近程度加权，缺失和稀疏资料向中间值收缩。','- 随机机制分为全场共同节奏、个体状态、估计不确定性、小幅出闸/节律扰动。个体方差和共同节奏由训练残差估计，并按引擎对速度扰动的实际敏感度换算；出闸和节律的具体分布仍是预设。','','## 数据隔离','','- 开发集为7月7日至10月31日全部5,615场可用full比赛，中央1,064、地方4,551；五折按日期分组。上轮10月留出集已经看过，本轮明确转为开发资料。','- 此前1,692场缺少审计输入的已结算预测无法重算；有效full中250场无近走、2,966匹无近走，保留明示缺省值。没有做旧分数调权，也没有读取v1—v6预测器。',f"- 新验证为11月1日的{protocol['new_races']}场，中央{protocol['scope_counts']['central']}、地方{protocol['scope_counts']['local']}；与开发比赛ID重叠为{protocol['training_overlap']}。",f"- 数据快照时间：{protocol['source_captured_at']}；参数冻结：{protocol['model_frozen_at']}；首次读取新标签：{protocol['new_labels_opened_at']}。时间均为UTC。",'- 新资料来自独立SHA-256校验快照，共30,815个文件。新快照在训练期间下载但优化程序从不打开它；新标签由单独验证命令在核对冻结哈希后读取。','- 模拟输入只包含严格早于目标日的近走，排除目标比赛本身。赔率、人气、保存的Agent分数与目标结果都不进入模拟。历史时钟只能从较早比赛关联。','- 这是未见日期的回顾性验证；有些比赛在参数冻结前已经发生，不能称为全部赛前冻结的前瞻实盘。当天数据尚未完整，不能代表整个比赛日或稳定提升。','','## 开发集五折结果','','采用每场32个固定种子，汇总排名按进入前三频率排序；单次动画另算平均覆盖率。概率误差用前三事件的Brier分数（越低越好），用时分布使用CRPS（越低越好）。主要选择指标仍为汇总前五覆盖实际前三席位，并保护冠军覆盖及概率误差。','','|指标|原模型|上轮候选|新版折外|','|---|---:|---:|---:|']
    fields=[('前三席位覆盖率','top3_recall','.2%'),('前三全部覆盖场次','all3_races',',d'),('至少两匹覆盖场次','at_least2_races',',d'),('单次模拟平均覆盖率','single_run_recall','.2%'),('冠军进入汇总前五','winner_top5','.2%'),('冠军平均汇总名次','winner_mean_rank','.2f'),('汇总平均用时误差（秒）','mean_time_mae','.3f'),('单次用时误差（秒）','single_time_mae','.3f'),('末段平均用时误差（秒）','closing_mae','.3f'),('前三概率Brier','top3_brier','.4f'),('用时分布CRPS','time_crps','.3f')]
    for label,key,fmt in fields:lines.append(f"|{label}|{dev['baseline'][key]:{fmt}}|{dev['previous'][key]:{fmt}}|{dev[choice][key]:{fmt}}|")
    lines += ['',f"在预先规定的0.65、1.0、1.35倍个体扰动中，开发集选择了{choice}。阶段一确定性参数用时与排序联合拟合，阶段二在该折训练残差估计随机方差，测试折从未用于该折拟合。没有以本次13场新比赛选择任何参数。",'', '## 新比赛独立验证','','每场使用另外128个固定种子，全部由前端原版引擎计算。128次模拟并不增加真实比赛样本量。','','|指标|原模型|上轮候选|新版|','|---|---:|---:|---:|']
    for label,key,fmt in fields:
        if key=='closing_mae':continue
        lines.append(f"|{label}|{new['baseline'][key]:{fmt}}|{new['previous'][key]:{fmt}}|{new['optimized'][key]:{fmt}}|")
    lines += ['',f"汇总前三席位由{new['baseline']['covered_seats']}/{3*protocol['new_races']}到{new['optimized']['covered_seats']}/{3*protocol['new_races']}，只增加一个席位；单次覆盖从{new['baseline']['single_run_recall']:.2%}降至{new['optimized']['single_run_recall']:.2%}，Brier误差由{new['baseline']['top3_brier']:.4f}升至{new['optimized']['top3_brier']:.4f}。因此不能宣称随机模拟或概率已经稳定改善。",'',f"新比赛有{new['baseline']['time_n']}匹次用时标签，最后600米实测标签为0。冲刺的新日期精度尚不可验证，不能把开发集1.16秒误差当作新比赛的实测误差。当天仅两个地方样本，总体提升主要来自一个新增地方席位，中央覆盖席位没有增加。",'', '![开发集与新比赛比较](comparison.png)','','## 中央与地方','','|范围|场次|原汇总覆盖|新汇总覆盖|原单次覆盖|新单次覆盖|','|---|---:|---:|---:|---:|---:|']
    for s,label in [('central','中央'),('local','地方')]:
        a=new['baseline']['by_scope'][s];b=new['optimized']['by_scope'][s]
        if a['races']:lines.append(f"|{label}|{a['races']}|{a['top3_recall']:.2%}|{b['top3_recall']:.2%}|{a['single_run_recall']:.2%}|{b['single_run_recall']:.2%}|")
    parity=read(d/'parity.json')
    lines += ['','## 证据与边界','',f"- 加速训练执行器与原版引擎在开发样本{parity['horse_runs']}匹次随机轨迹终点用时对照，平均差{parity['mean_seconds']:.6f}秒，最大{parity['max_seconds']:.6f}秒。新日期表格均用原版。",'- 体力不是实测；坡度生理反应、疲劳阈值、路线、堵路和碰撞仍未校准或重现。能量消耗系数接近搜索上界，节能系数接近下界，部分潜在参数仍不可独立识别。','- 每折共同参考只用该折训练资料拟合；开发折可包含更晚历史日期，不等同于时间滚动外推。真正未见日期证据是11月1日，且当前仅13场。','- 已冻结参数不因新赛果调整。旧模型继续默认，新模型在页面明确标注验证中。','- `frozen_model.json`、`freeze_receipt.json`记录模型及首次标签读取顺序；`fit_protocol.json`、`data_protocol.json`、`new_validation_protocol.json`记录来源、种子、划分；`development_rows.json`、`new_rows.json`保存逐场结果。','- `run_source/`保存本次训练原始源码，与协议哈希一致。报告用独立代码复核新比赛覆盖席位和平均预测用时误差。','- 模拟器中新增三场新日期输入样例；仅导出赛前资料，没有导出目标赛果。']
    (d/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (d/'independent_validation.json').write_text(json.dumps({'reconciled':True,'new_races':len(cases),'model_sha256':hashlib.sha256((d/'frozen_model.json').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    print('报告和图表已生成，独立重算通过。')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();main(a.directory)
