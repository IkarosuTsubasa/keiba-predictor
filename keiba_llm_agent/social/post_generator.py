from __future__ import annotations

from pathlib import Path

from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import Review


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RACE_DATA_DIR = BASE_DIR / "data" / "race_data"
DEFAULT_RESULTS_DIR = BASE_DIR / "data" / "results"
HASHTAGS = "#いかいもAI競馬 #競馬"
MAX_POST_LENGTH = 280
FALLBACK_NOTE = "※暫定ロジックによる試験運用"


def _load_default_race_data(race_id: str) -> RaceData | None:
    path = DEFAULT_RACE_DATA_DIR / f"{race_id}.json"
    if not path.exists():
        return None
    return RaceData.from_json_file(path)


def _load_default_result_data(race_id: str) -> ResultData | None:
    path = DEFAULT_RESULTS_DIR / f"{race_id}.json"
    if not path.exists():
        return None
    return ResultData.model_validate_json(path.read_text(encoding="utf-8"))


def _horse_name_map(prediction: Prediction, race_data: RaceData | None = None) -> dict[int, str]:
    mapping = {horse_score.horse_no: horse_score.horse_name for horse_score in prediction.horse_scores}
    if race_data is not None:
        for horse in race_data.horses:
            mapping.setdefault(horse.horse_no, horse.horse_name)
    return mapping


def _mark_lines(prediction: Prediction, race_data: RaceData | None = None) -> list[str]:
    horse_names = _horse_name_map(prediction, race_data)
    lines: list[str] = []
    for mark in ("◎", "○", "▲", "△", "☆"):
        horse_no = prediction.marks.get(mark, 0)
        horse_name = horse_names.get(horse_no, "unknown")
        lines.append(f"{mark}{horse_no} {horse_name}")
    return lines


def _bets_line(prediction: Prediction) -> str:
    if not prediction.bets:
        return "買い目：なし"
    bet_parts = []
    for bet in prediction.bets[:2]:
        horses = "-".join(str(number) for number in bet.horse_numbers)
        bet_parts.append(f"{bet.bet_type} {horses}")
    return "買い目：" + " / ".join(bet_parts)


def _compact_reason(text: str, limit: int = 36) -> str:
    compact = " ".join(text.replace("\n", " ").split())
    if len(compact) <= limit:
        return compact
    for delimiter in ("。", "、"):
        head = compact.split(delimiter)[0].strip()
        if head and len(head) <= limit:
            return head + ("。" if delimiter == "。" else "")
    return compact[:limit].rstrip()


def _finalize_post(lines: list[str]) -> str:
    return "\n".join(lines).strip() + "\n"


def _truncate_post(lines: list[str], max_length: int = MAX_POST_LENGTH) -> str:
    removable_prefixes = [
        "血統補正もプラス。",
        "血統面では",
        "相手関係では",
        "シミュレーションでは",
        "展開シミュレーションは",
        "展開想定は",
        "深掘り分析では",
    ]
    trimmed = list(lines)
    text = _finalize_post(trimmed)
    if len(text) <= max_length:
        return text

    for prefix in removable_prefixes:
        text = _finalize_post(trimmed)
        if len(text) <= max_length:
            return text
        trimmed = [line for line in trimmed if not line.startswith(prefix)]

    text = _finalize_post(trimmed)
    if len(text) <= max_length:
        return text

    optional_prefixes = [
        "lesson：",
        "※heuristic",
        "根拠：",
    ]
    for prefix in optional_prefixes:
        text = _finalize_post(trimmed)
        if len(text) <= max_length:
            return text
        trimmed = [line for line in trimmed if not line.startswith(prefix)]

    text = _finalize_post(trimmed)
    if len(text) <= max_length:
        return text

    # 最后仅保留高优先级核心信息，不做半句截断。
    keep_lines: list[str] = []
    for line in trimmed:
        if (
            not line
            or line == HASHTAGS
            or line.endswith("予想")
            or line.startswith(("◎", "○", "▲", "△", "☆", "買い判断：", "買い目："))
            or line == FALLBACK_NOTE
        ):
            keep_lines.append(line)
    text = _finalize_post(keep_lines)
    if len(text) <= max_length:
        return text
    return _finalize_post([line for line in keep_lines if line != FALLBACK_NOTE])


def _normalize_simulation_line(text: str | None) -> str | None:
    if not text:
        return None
    compact = " ".join(text.split())
    if compact.startswith("シミュレーションでは"):
        return compact
    return f"シミュレーションでは{compact}"


def build_prediction_post(prediction: Prediction, race_data: RaceData | None = None) -> str:
    race_name = prediction.race_info.race_name if prediction.race_info and prediction.race_info.race_name else prediction.race_id
    strategy = prediction.strategy
    confidence = strategy.confidence if strategy is not None else "unknown"
    bet_decision = strategy.bet_decision if strategy is not None else "unknown"
    reason_source = strategy.reason if strategy is not None else prediction.summary
    deep_line = None
    pedigree_line = None
    race_level_line = None
    pace_line = None
    simulation_line = None
    if prediction.deep_analyses:
        top_horse_no = prediction.marks.get("◎", 0)
        top_analysis = next(
            (analysis for analysis in prediction.deep_analyses if analysis.horse_no == top_horse_no),
            None,
        )
        if top_analysis is not None:
            key_points = top_analysis.positive_flags[:2]
            if key_points:
                translated = "＋".join(
                    {
                        "RECENT_FORM_STRONG": "近走好調",
                        "RECENT_FORM_STABLE": "近走安定",
                        "DISTANCE_FIT": "距離適性",
                        "COURSE_FIT": "コース適性",
                        "TRACK_CONDITION_FIT": "馬場適性",
                        "JOCKEY_CONTINUITY": "騎手継続",
                        "VALUE_CANDIDATE": "妙味",
                        "STABLE_PERFORMER": "堅実さ",
                    }.get(flag, flag)
                    for flag in key_points
                )
                deep_line = f"深掘り分析では◎は{translated}を評価。"
    if prediction.pedigree_analyses:
        top_horse_no = prediction.marks.get("◎", 0)
        top_pedigree = next(
            (analysis for analysis in prediction.pedigree_analyses if analysis.horse_no == top_horse_no),
            None,
        )
        if top_pedigree is not None and top_pedigree.distance_tendency != "unknown":
            pedigree_line = f"血統面では◎は{_compact_reason(top_pedigree.distance_tendency, limit=16)}適性も評価。"
    if prediction.race_level_analyses:
        top_horse_no = prediction.marks.get("◎", 0)
        top_race_level = next(
            (analysis for analysis in prediction.race_level_analyses if analysis.horse_no == top_horse_no),
            None,
        )
        if top_race_level is not None:
            if "HEAD_TO_HEAD_NEGATIVE" in top_race_level.risk_flags:
                race_level_line = "相手関係では◎にやや減点。"
            else:
                race_level_line = "相手関係では◎に大きな減点なし。"
    if prediction.race_pace_projection is not None:
        pace_map = {
            "slow": "スロー",
            "average": "平均",
            "fast": "ハイ",
            "unknown": "不明",
        }
        styles = "・".join(prediction.race_pace_projection.favorable_styles[:2]) if prediction.race_pace_projection.favorable_styles else "不明"
        pace_line = f"展開想定は{pace_map.get(prediction.race_pace_projection.projected_pace, prediction.race_pace_projection.projected_pace)}ペース、{styles}勢を評価。"
    if prediction.race_simulation is not None and prediction.race_simulation.reasoning_summary:
        simulation_line = _normalize_simulation_line(prediction.race_simulation.reasoning_summary)
    pedigree_bonus_line = None
    top_score = next((score for score in prediction.horse_scores if score.horse_no == prediction.marks.get("◎", 0)), None)
    if top_score is not None and top_score.pedigree_adjustment.pedigree_adjustment > 0:
        pedigree_bonus_line = "血統補正もプラス。"
    lines = [
        f"{race_name} 予想",
        "",
        *_mark_lines(prediction, race_data),
        "",
        f"買い判断：{bet_decision} / confidence={confidence}",
        _bets_line(prediction),
        "",
        _compact_reason(reason_source, limit=40),
        *( [deep_line] if deep_line else [] ),
        *( [pace_line] if pace_line else [] ),
        *( [simulation_line] if simulation_line else [] ),
        *( [race_level_line] if race_level_line else [] ),
        *( [pedigree_line] if pedigree_line else [] ),
        *( [pedigree_bonus_line] if pedigree_bonus_line else [] ),
        FALLBACK_NOTE,
        "",
        HASHTAGS,
    ]
    return _truncate_post(lines)


def build_review_post(
    prediction: Prediction,
    review: Review,
    result_data: ResultData | None = None,
    race_data: RaceData | None = None,
) -> str:
    race_name = prediction.race_info.race_name if prediction.race_info and prediction.race_info.race_name else prediction.race_id
    horse_names = _horse_name_map(prediction, race_data)
    top3_text = "結果：unknown"
    if result_data is not None:
        top3_text = f"結果：{result_data.result.first}→{result_data.result.second}→{result_data.result.third}"
    top_mark = prediction.marks.get("◎", 0)
    top_mark_name = horse_names.get(top_mark, "unknown")
    first_bet_result = review.bet_results[0] if review.bet_results else None
    if first_bet_result is not None:
        bet_horses = "-".join(str(number) for number in first_bet_result.horse_numbers)
        bet_line = f"買い目：{first_bet_result.bet_type}{bet_horses} {'的中' if first_bet_result.hit else '不的中'}"
    else:
        bet_line = "買い目：なし"
    lesson_text = review.lessons[0].lesson if review.lessons else "なし"
    roi_percent = int(review.hit_summary.roi * 100) if review.hit_summary.total_stake > 0 else 0
    simulation_line = None
    if review.simulation_review is not None:
        compact = _compact_reason(review.simulation_review.pace_prediction_review, limit=26)
        simulation_line = f"展開シミュレーションは{compact}"
    lines = [
        f"{race_name} 回顧",
        "",
        top3_text,
        f"印内Top3：{review.hit_summary.marked_horses_top3_count}頭",
        f"◎{top_mark} {top_mark_name}は{'3着内' if review.hit_summary.main_mark_top3 else '圏外'}",
        "",
        bet_line,
        f"回収率：{roi_percent}%",
        *( [simulation_line] if simulation_line else [] ),
        "",
        f"lesson：{_compact_reason(lesson_text, limit=34)}",
        "",
        HASHTAGS,
    ]
    return _truncate_post(lines)


def generate_prediction_post(prediction: Prediction) -> str:
    return build_prediction_post(prediction, race_data=_load_default_race_data(prediction.race_id))


def generate_review_post(prediction: Prediction, review: Review) -> str:
    return build_review_post(
        prediction,
        review,
        result_data=_load_default_result_data(prediction.race_id),
        race_data=_load_default_race_data(prediction.race_id),
    )


def save_post(text: str, output_path: str | Path) -> Path:
    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(text, encoding="utf-8")
    return final_path
