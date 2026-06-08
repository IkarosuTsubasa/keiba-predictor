from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from keiba_llm_agent.memory.lesson_store import LessonStore
from keiba_llm_agent.parsers.netkeiba_race_parser import infer_scope_key, infer_source_from_race_id
from keiba_llm_agent.schemas.prediction import Prediction
from keiba_llm_agent.schemas.race_data import RaceInfo
from keiba_llm_agent.schemas.result import ResultData
from keiba_llm_agent.schemas.review import LessonItem, Review


HASHTAGS = "#いかいもAI競馬 #競馬"
MAX_POST_LENGTH = 280
VALID_SCOPE_KEYS = {"central", "central_turf", "central_dirt", "local"}
SCOPE_LABELS_JA = {
    "": "全場",
    "central": "中央",
    "central_turf": "中央芝",
    "central_dirt": "中央ダート",
    "local": "地方",
    "unknown": "不明",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_yen(value: int) -> str:
    return f"{value}円"


def _normalize_scope_key(scope_key: str | None) -> str | None:
    text = str(scope_key or "").strip().lower()
    if not text or text in {"all", "any"}:
        return None
    if text not in VALID_SCOPE_KEYS:
        raise ValueError(f"invalid scope_key: {scope_key}")
    return text


def _scope_label(scope_key: str | None) -> str:
    text = str(scope_key or "").strip().lower()
    return SCOPE_LABELS_JA.get(text, text or SCOPE_LABELS_JA[""])


def _prediction_scope_key(prediction: Prediction) -> str | None:
    race_info = prediction.race_info
    explicit_scope = str(getattr(race_info, "scope_key", "") or "").strip().lower() if race_info else ""
    if explicit_scope:
        return explicit_scope

    explicit_source = str(getattr(race_info, "source", "") or "").strip().lower() if race_info else ""
    if explicit_source == "local":
        return "local"

    inferred_scope = infer_scope_key(
        prediction.race_id,
        surface=getattr(race_info, "surface", None) if race_info else None,
        course=getattr(race_info, "course", None) if race_info else None,
    )
    if inferred_scope:
        return inferred_scope

    inferred_source = infer_source_from_race_id(prediction.race_id)
    if inferred_source == "local":
        return "local"
    if explicit_source == "central" or inferred_source == "central":
        return "central"
    surface = str(getattr(race_info, "surface", "") or "").strip() if race_info else ""
    if surface == "芝":
        return "central_turf"
    if surface and ("ダ" in surface or "砂" in surface):
        return "central_dirt"
    return None


def _prediction_matches_scope(prediction: Prediction, scope_key: str | None) -> bool:
    normalized_scope = _normalize_scope_key(scope_key)
    if normalized_scope is None:
        return True
    prediction_scope = _prediction_scope_key(prediction)
    if normalized_scope == "central":
        return prediction_scope in {"central", "central_turf", "central_dirt"}
    return prediction_scope == normalized_scope


def _truncate_post(text: str) -> str:
    if len(text) <= MAX_POST_LENGTH:
        return text
    return text[: MAX_POST_LENGTH - 1].rstrip() + "…"


def _load_predictions_for_date(
    predictions_dir: Path,
    target_date: str,
    scope_key: str | None = None,
) -> tuple[list[Prediction], list[str]]:
    predictions: list[Prediction] = []
    warnings: list[str] = []
    normalized_scope = _normalize_scope_key(scope_key)
    if not predictions_dir.exists():
        return predictions, warnings

    for path in sorted(predictions_dir.glob("*.json")):
        prediction = Prediction.model_validate_json(path.read_text(encoding="utf-8"))
        race_info = prediction.race_info
        if race_info is None or not race_info.race_date:
            warnings.append(
                f"prediction skipped because race_info.race_date is missing: race_id={prediction.race_id}"
            )
            continue
        if race_info.race_date == target_date and _prediction_matches_scope(prediction, normalized_scope):
            predictions.append(prediction)
    return predictions, warnings


def _load_optional_review(reviews_dir: Path, race_id: str) -> Review | None:
    path = reviews_dir / f"{race_id}.json"
    if not path.exists():
        return None
    return Review.model_validate_json(path.read_text(encoding="utf-8"))


def _load_optional_result(results_dir: Path, race_id: str) -> ResultData | None:
    path = results_dir / f"{race_id}.json"
    if not path.exists():
        return None
    return ResultData.model_validate_json(path.read_text(encoding="utf-8"))


def _load_lessons_for_race_ids(lessons_path: Path, race_ids: set[str]) -> list[LessonItem]:
    if not lessons_path.exists():
        return []
    payload = json.loads(lessons_path.read_text(encoding="utf-8"))
    lessons = [LessonStore.normalize_lesson(item) for item in payload]
    return [
        lesson
        for lesson in lessons
        if lesson.enabled and lesson.source_race_id in race_ids
    ]


def _horse_name_by_number(prediction: Prediction) -> dict[int, str]:
    return {
        horse_score.horse_no: horse_score.horse_name
        for horse_score in prediction.horse_scores
    }


def _main_mark_text(prediction: Prediction) -> str:
    horse_no = prediction.marks.get("◎", 0)
    if horse_no <= 0:
        return "-"
    horse_name = _horse_name_by_number(prediction).get(horse_no, "")
    return f"{horse_no} {horse_name}".strip()


def _result_text(result_data: ResultData | None, horse_names: dict[int, str]) -> str:
    if result_data is None:
        return "-"
    top3 = [
        (result_data.result.first, "1着"),
        (result_data.result.second, "2着"),
        (result_data.result.third, "3着"),
    ]
    parts: list[str] = []
    for horse_no, _ in top3:
        name = horse_names.get(horse_no, "")
        label = f"{horse_no} {name}".strip()
        parts.append(label)
    return " / ".join(parts)


def _lesson_text(lesson: LessonItem) -> str:
    return (
        f"{lesson.course}/{lesson.surface}/{lesson.distance}m/{lesson.track_condition} "
        f"[{lesson.confidence} / score={lesson.score:.2f}] {lesson.lesson}"
    )


def _collect_new_lessons(
    reviews: list[Review],
    lessons_path: Path,
) -> list[LessonItem]:
    reviewed_race_ids = {review.race_id for review in reviews}
    loaded_lessons = _load_lessons_for_race_ids(lessons_path, reviewed_race_ids)
    if not loaded_lessons:
        loaded_lessons = [lesson for review in reviews for lesson in review.lessons if lesson.enabled]

    deduped: list[LessonItem] = []
    seen_source_race_ids: set[str] = set()
    for lesson in loaded_lessons:
        if lesson.source_race_id in seen_source_race_ids:
            continue
        seen_source_race_ids.add(lesson.source_race_id)
        deduped.append(lesson)
    return deduped


def _bet_rows_for_race(
    prediction: Prediction,
    review: Review | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if review is not None and review.bet_results:
        for item in review.bet_results:
            rows.append(
                {
                    "race_id": prediction.race_id,
                    "race_name": prediction.race_info.race_name if prediction.race_info else prediction.race_id,
                    "bet_type": item.bet_type,
                    "horse_numbers": item.horse_numbers,
                    "amount": item.amount,
                    "hit": item.hit,
                    "return_amount": item.return_amount,
                }
            )
        return rows

    for bet in prediction.bets:
        rows.append(
            {
                "race_id": prediction.race_id,
                "race_name": prediction.race_info.race_name if prediction.race_info else prediction.race_id,
                "bet_type": bet.bet_type,
                "horse_numbers": bet.horse_numbers,
                "amount": bet.amount or 0,
                "hit": None,
                "return_amount": None,
            }
        )
    return rows


def build_daily_context(
    target_date: str,
    predictions_dir: Path,
    reviews_dir: Path,
    results_dir: Path,
    lessons_path: Path,
    scope_key: str | None = None,
) -> dict[str, object]:
    normalized_scope = _normalize_scope_key(scope_key)
    predictions, warnings = _load_predictions_for_date(predictions_dir, target_date, scope_key=normalized_scope)
    race_rows: list[dict[str, object]] = []
    bet_rows: list[dict[str, object]] = []
    pending_races: list[dict[str, str]] = []
    reviews: list[Review] = []
    good_point_counter: Counter[str] = Counter()
    bad_point_counter: Counter[str] = Counter()

    total_stake = 0
    total_return = 0
    bet_race_count = 0
    hit_bet_count = 0
    reviewed_race_count = 0
    main_mark_top3_hits = 0
    marked_top3_total = 0
    payout_warning_count = 0
    scope_counter: Counter[str] = Counter()

    for prediction in predictions:
        race_info: RaceInfo | None = prediction.race_info
        prediction_scope = _prediction_scope_key(prediction) or "unknown"
        scope_counter.update([prediction_scope])
        horse_names = _horse_name_by_number(prediction)
        review = _load_optional_review(reviews_dir, prediction.race_id)
        result_data = _load_optional_result(results_dir, prediction.race_id)
        has_bets = (
            (prediction.strategy is not None and prediction.strategy.bet_decision == "BET")
            or bool(prediction.bets)
        )
        if has_bets:
            bet_race_count += 1

        if review is not None:
            reviews.append(review)
            reviewed_race_count += 1
            total_stake += review.hit_summary.total_stake
            total_return += review.hit_summary.total_return
            if review.payout_warning:
                payout_warning_count += 1
            main_mark_top3_hits += 1 if review.hit_summary.main_mark_top3 else 0
            marked_top3_total += review.hit_summary.marked_horses_top3_count
            if has_bets and review.hit_summary.bet_hit:
                hit_bet_count += 1
            good_point_counter.update(review.good_points)
            bad_point_counter.update(review.bad_points)
        else:
            pending_races.append(
                {
                    "race_id": prediction.race_id,
                    "race_name": race_info.race_name if race_info else prediction.race_id,
                }
            )

        race_rows.append(
            {
                "race_id": prediction.race_id,
                "scope_key": prediction_scope,
                "scope_label": _scope_label(prediction_scope),
                "race_name": race_info.race_name if race_info else prediction.race_id,
                "bet_decision": prediction.strategy.bet_decision if prediction.strategy else "unknown",
                "confidence": prediction.strategy.confidence if prediction.strategy else "unknown",
                "main_mark": _main_mark_text(prediction),
                "result_text": _result_text(result_data, horse_names),
                "marked_top3": review.hit_summary.marked_horses_top3_count if review is not None else None,
                "bet_hit": review.hit_summary.bet_hit if review is not None else None,
                "stake": review.hit_summary.total_stake if review is not None else 0,
                "return_amount": review.hit_summary.total_return if review is not None else 0,
                "roi": review.hit_summary.roi if review is not None else None,
                "payout_warning": review.payout_warning if review is not None else False,
                "status": "reviewed" if review is not None else "pending",
            }
        )
        bet_rows.extend(_bet_rows_for_race(prediction, review))

    reviewed_bet_denominator = bet_race_count if bet_race_count > 0 else 0
    roi = (total_return / total_stake) if total_stake > 0 else 0.0
    main_mark_top3_rate = (
        main_mark_top3_hits / reviewed_race_count if reviewed_race_count > 0 else 0.0
    )
    marked_top3_rate = (
        marked_top3_total / (reviewed_race_count * 3) if reviewed_race_count > 0 else 0.0
    )
    bet_hit_rate = (
        hit_bet_count / reviewed_bet_denominator if reviewed_bet_denominator > 0 else 0.0
    )
    roi_reliable = payout_warning_count == 0
    if not roi_reliable:
        warnings.append("Some payout data missing. ROI may be unreliable.")
    lessons = _collect_new_lessons(reviews, lessons_path)

    return {
        "date": target_date,
        "scope_key": normalized_scope,
        "scope_label": _scope_label(normalized_scope),
        "warnings": warnings,
        "race_rows": race_rows,
        "bet_rows": bet_rows,
        "pending_races": pending_races,
        "good_points": [item for item, _ in good_point_counter.most_common(10)],
        "bad_points": [item for item, _ in bad_point_counter.most_common(10)],
        "lessons": lessons,
        "metrics": {
            "scope_counts": dict(scope_counter),
            "target_race_count": len(predictions),
            "predicted_race_count": len(predictions),
            "reviewed_race_count": reviewed_race_count,
            "bet_race_count": bet_race_count,
            "hit_bet_count": hit_bet_count,
            "total_stake": total_stake,
            "total_return": total_return,
            "roi": roi,
            "roi_reliable": roi_reliable,
            "main_mark_top3_rate": main_mark_top3_rate,
            "marked_top3_rate": marked_top3_rate,
            "bet_hit_rate": bet_hit_rate,
        },
    }


def generate_daily_report(
    target_date: str,
    predictions_dir: Path,
    reviews_dir: Path,
    results_dir: Path,
    lessons_path: Path,
    scope_key: str | None = None,
) -> tuple[str, dict[str, object]]:
    context = build_daily_context(
        target_date=target_date,
        predictions_dir=predictions_dir,
        reviews_dir=reviews_dir,
        results_dir=results_dir,
        lessons_path=lessons_path,
        scope_key=scope_key,
    )
    metrics = context["metrics"]
    race_rows = context["race_rows"]
    bet_rows = context["bet_rows"]
    lessons: list[LessonItem] = context["lessons"]
    warnings: list[str] = context["warnings"]
    pending_races: list[dict[str, str]] = context["pending_races"]
    scope_counts = dict(metrics.get("scope_counts", {}) or {})
    scope_breakdown = " / ".join(
        f"{_scope_label(key)} {value}R"
        for key, value in sorted(scope_counts.items(), key=lambda item: (_scope_label(item[0]), item[0]))
    ) or "-"

    lines = [
        f"# {target_date} AI競馬 Daily Summary",
        "",
        "## サマリ",
        f"- 日付: {target_date}",
        f"- 対象スコープ: {context['scope_label']}",
        f"- スコープ内訳: {scope_breakdown}",
        f"- 対象レース数: {metrics['target_race_count']}",
        f"- 予想済みレース数: {metrics['predicted_race_count']}",
        f"- 回顧済みレース数: {metrics['reviewed_race_count']}",
        f"- BETレース数: {metrics['bet_race_count']}",
        f"- 的中BET数: {metrics['hit_bet_count']}",
        f"- total_stake: {metrics['total_stake']}",
        f"- total_return: {metrics['total_return']}",
        f"- ROI: {_format_rate(metrics['roi'])}{' (暫定)' if not metrics['roi_reliable'] else ''}",
        f"- ◎ Top3率: {_format_rate(metrics['main_mark_top3_rate'])}",
        f"- 印内Top3率: {_format_rate(metrics['marked_top3_rate'])}",
        f"- bet_hit率: {_format_rate(metrics['bet_hit_rate'])}",
        "",
        "## レース別結果",
        "| scope | race_id | race_name | bet_decision | confidence | ◎ | 結果 | 印内Top3 | bet_hit | stake | return | ROI |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]

    for row in race_rows:
        marked_top3 = "-" if row["marked_top3"] is None else str(row["marked_top3"])
        bet_hit = "-"
        if row["bet_decision"] == "SKIP" or (row["status"] == "reviewed" and row["stake"] == 0):
            bet_hit = "見送り"
        elif row["bet_hit"] is True:
            bet_hit = "的中"
        elif row["bet_hit"] is False:
            bet_hit = "不的中"
        roi_text = "-" if row["roi"] is None else _format_rate(row["roi"])
        if row["payout_warning"] and roi_text != "-":
            roi_text = f"{roi_text} (暫定)"
        lines.append(
            f"| {row['scope_label']} | {row['race_id']} | {row['race_name']} | {row['bet_decision']} | {row['confidence']} | "
            f"{row['main_mark']} | {row['result_text']} | {marked_top3} | {bet_hit} | "
            f"{row['stake']} | {row['return_amount']} | {roi_text} |"
        )

    lines.extend(
        [
            "",
            "## BET一覧",
            "| scope | race_id | race_name | bet_type | horse_numbers | amount | hit | return_amount |",
            "| --- | --- | --- | --- | --- | ---: | --- | ---: |",
        ]
    )
    if bet_rows:
        scope_by_race_id = {str(row["race_id"]): str(row.get("scope_label", "")) for row in race_rows}
        for row in bet_rows:
            hit = "pending"
            if row["hit"] is True:
                hit = "的中"
            elif row["hit"] is False:
                hit = "不的中"
            return_amount = "-" if row["return_amount"] is None else str(row["return_amount"])
            horse_numbers = "-".join(str(number) for number in row["horse_numbers"])
            lines.append(
                f"| {scope_by_race_id.get(str(row['race_id']), '-')} | {row['race_id']} | {row['race_name']} | {row['bet_type']} | {horse_numbers} | "
                f"{row['amount']} | {hit} | {return_amount} |"
            )
    else:
        lines.append("| - | - | - | - | - | 0 | - | 0 |")

    lines.extend(["", "## 良かった点"])
    if context["good_points"]:
        lines.extend(f"- {item}" for item in context["good_points"])
    else:
        lines.append("- なし")

    lines.extend(["", "## 反省点"])
    if context["bad_points"]:
        lines.extend(f"- {item}" for item in context["bad_points"])
    else:
        lines.append("- なし")

    lines.extend(["", "## 新規 lessons"])
    if lessons:
        lines.extend(f"- {_lesson_text(lesson)}" for lesson in lessons)
    else:
        lines.append("- なし")

    lines.extend(["", "## 未回顧レース"])
    if pending_races:
        lines.extend(
            f"- {item['race_id']} {item['race_name']}".strip()
            for item in pending_races
        )
    else:
        lines.append("- なし")

    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines) + "\n", context


def generate_daily_social_post(context: dict[str, object]) -> str:
    metrics = context["metrics"]
    lessons: list[LessonItem] = context["lessons"]
    bad_points: list[str] = context["bad_points"]

    lesson_text = "なし"
    if lessons:
        lesson_text = lessons[0].lesson
    elif bad_points:
        lesson_text = bad_points[0]

    text = (
        f"{context['date']} AI競馬まとめ\n\n"
        f"対象: {context.get('scope_label', '全場')} {metrics['target_race_count']}レース\n"
        f"BET: {metrics['bet_race_count']}レース / 的中: {metrics['hit_bet_count']}\n"
        f"投資: {_format_yen(metrics['total_stake'])}\n"
        f"回収: {_format_yen(metrics['total_return'])}\n"
        f"回収率: {metrics['roi'] * 100:.0f}%\n\n"
        f"◎Top3率: {metrics['main_mark_top3_rate'] * 100:.0f}%\n"
        f"印内Top3率: {metrics['marked_top3_rate'] * 100:.0f}%\n\n"
        f"今日のlesson:\n「{lesson_text}」\n\n"
        f"{HASHTAGS}"
    )
    if len(text) <= MAX_POST_LENGTH:
        return text

    shorter_lesson = lesson_text[:24].rstrip() + "…" if len(lesson_text) > 24 else lesson_text
    text = (
        f"{context['date']} AI競馬まとめ\n\n"
        f"{context.get('scope_label', '全場')} 対象{metrics['target_race_count']}R / BET{metrics['bet_race_count']}R / 的中{metrics['hit_bet_count']}R\n"
        f"投資{metrics['total_stake']}円 / 回収{metrics['total_return']}円 / 回収率{metrics['roi'] * 100:.0f}%\n"
        f"◎Top3率{metrics['main_mark_top3_rate'] * 100:.0f}% / 印内Top3率{metrics['marked_top3_rate'] * 100:.0f}%\n"
        f"lesson: {shorter_lesson}\n\n"
        f"{HASHTAGS}"
    )
    return _truncate_post(text)


def save_daily_report(markdown: str, output_path: str | Path) -> Path:
    final_path = Path(output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(markdown, encoding="utf-8")
    return final_path
