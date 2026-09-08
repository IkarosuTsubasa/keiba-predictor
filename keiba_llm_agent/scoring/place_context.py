from __future__ import annotations

import numpy as np

from keiba_llm_agent.schemas.prediction import HorseScore
from keiba_llm_agent.schemas.race_data import RaceData


def build_place_context_matrix(
    race: RaceData, scores: list[HorseScore], native: np.ndarray,
) -> np.ndarray:
    # Import lazily: the scope feature builder dispatches here for this feature version.
    from keiba_llm_agent.scoring.scope_reranker import (
        RUN_FEATURE_NAMES, _parse_date, build_run_features,
    )

    rows = []
    race_date = _parse_date(race.race_info.race_date)
    score_map = {score.horse_no: score for score in scores}
    for horse in race.horses:
        runs = [run for run in horse.recent_runs[:8]
                if run.finish is not None and run.field_size is not None and run.field_size > 1][:5]
        features = build_run_features(horse, race)
        if not runs:
            features = {name: np.nan for name in RUN_FEATURE_NAMES}
            features["run_count"] = 0.0
        last = runs[0] if runs else None
        last_date = _parse_date(last.date) if last else None
        rest_missing = not (race_date and last_date)
        distance_missing = not (last and last.distance and race.race_info.distance)
        margin_missing = not last or last.margin is None
        corners_missing = not last or not last.corner_positions
        final3f_missing = sum(run.final_3f is not None for run in runs) < 2
        if rest_missing:
            features["rest_days_preference"] = np.nan
        if distance_missing:
            features["distance_change_penalty"] = np.nan
        if margin_missing:
            features["last_margin_closeness"] = np.nan
        if corners_missing:
            features["last_position_gain"] = np.nan
            features["last_early_position"] = np.nan
        if final3f_missing:
            features["last_final3f_vs_history"] = np.nan
        for name, predicate in (
            ("same_distance_best", lambda run: run.distance == race.race_info.distance and run.distance is not None),
            ("same_course_best", lambda run: run.course == race.race_info.course and bool(run.course)),
            ("same_surface_best", lambda run: run.surface == race.race_info.surface and bool(run.surface)),
            ("large_field_best", lambda run: run.field_size >= 13),
            ("same_jockey_best", lambda run: run.jockey == horse.jockey and bool(run.jockey)),
        ):
            if not any(predicate(run) for run in runs):
                features[name] = np.nan
        absolute = [
            float(len(race.horses)), float(len(race.horses) >= 13),
            float(race.race_info.distance or np.nan), float(len(horse.recent_runs)),
            float((race_date - last_date).days) if not rest_missing else np.nan,
            float(horse.frame_no) if horse.frame_no else np.nan,
            float(horse.carried_weight) if horse.carried_weight else np.nan,
            float(score_map[horse.horse_no].scores.ability_score),
            float(score_map[horse.horse_no].scores.recent_quality_score),
        ]
        missing = [float(not runs), float(rest_missing), float(distance_missing),
                   float(margin_missing), float(corners_missing), float(final3f_missing)]
        rows.append([features[name] for name in RUN_FEATURE_NAMES] + absolute + missing)
    return np.concatenate((native, np.asarray(rows, dtype=float)), axis=1)
