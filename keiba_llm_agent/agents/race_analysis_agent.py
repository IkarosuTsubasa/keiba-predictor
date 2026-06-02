from __future__ import annotations

from pathlib import Path

from keiba_llm_agent.config.scoring_config import (
    DEFAULT_SCORING_PROFILE,
    resolve_scoring_profile_config,
)
from keiba_llm_agent.llm import BaseLLMClient
from keiba_llm_agent.memory.lesson_store import LessonStore
from keiba_llm_agent.scoring.recent_run_scorer import (
    build_prediction_from_recent_runs_with_scoring_config,
    has_recent_runs_data,
)
from keiba_llm_agent.schemas.prediction import Prediction, ScoringConfigSnapshot
from keiba_llm_agent.schemas.race_data import RaceData
from keiba_llm_agent.schemas.review import LessonItem
from keiba_llm_agent.simulation.race_simulator import simulate_race


class RaceAnalysisAgent:
    def __init__(self, llm_client: BaseLLMClient, prompt_path: str | Path | None = None) -> None:
        default_prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "analysis_prompt.txt"
        self.llm_client = llm_client
        self.prompt_path = Path(prompt_path) if prompt_path else default_prompt_path
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")

    @staticmethod
    def _attach_simulation_summary(prediction: Prediction) -> None:
        if prediction.race_simulation is None:
            return
        summary_text = prediction.race_simulation.reasoning_summary.strip()
        if summary_text and summary_text not in prediction.summary:
            prediction.summary = f"{prediction.summary} {summary_text}".strip()
        for warning_text in prediction.race_simulation.warnings:
            if warning_text not in prediction.risks:
                prediction.risks.append(warning_text)

    def run(
        self,
        race_data: RaceData,
        lessons: list[LessonItem],
        scoring_profile: str = DEFAULT_SCORING_PROFILE,
        scoring_config: ScoringConfigSnapshot | None = None,
        borderline_recovery_enabled: bool = False,
    ) -> Prediction:
        if scoring_config is None:
            resolved_profile, _ = resolve_scoring_profile_config(scoring_profile=scoring_profile)
            scoring_profile = resolved_profile.scoring_profile
            scoring_config = ScoringConfigSnapshot.model_validate(
                resolved_profile.scoring_config.model_dump()
            )
            borderline_recovery_enabled = resolved_profile.borderline_recovery_enabled
        relevant_lessons = LessonStore.filter_relevant_lessons(
            lessons,
            race_data.race_info,
            current_race_id=race_data.race_info.race_id,
        )
        if has_recent_runs_data(race_data):
            prediction = build_prediction_from_recent_runs_with_scoring_config(
                race_data,
                relevant_lessons,
                scoring_profile=scoring_profile,
                scoring_config=scoring_config,
                borderline_recovery_enabled=borderline_recovery_enabled,
            )
            prediction.race_simulation = simulate_race(
                race_info=race_data.race_info,
                marks=prediction.marks,
                horse_scores=prediction.horse_scores,
                bets=prediction.bets,
                deep_analyses=prediction.deep_analyses,
                pedigree_analyses=prediction.pedigree_analyses,
                race_level_analyses=prediction.race_level_analyses,
                pace_analyses=prediction.pace_analyses,
                race_pace_projection=prediction.race_pace_projection,
                strategy=prediction.strategy,
                llm_client=self.llm_client,
            )
            enhancement = self.llm_client.enhance_prediction(
                prediction=prediction,
                race_data=race_data,
                used_lessons=relevant_lessons,
            )
            if enhancement:
                summary = enhancement.get("summary")
                risks = enhancement.get("risks")
                commentary = enhancement.get("commentary")
                if isinstance(summary, str) and summary.strip():
                    prediction.summary = summary
                if isinstance(risks, list) and all(isinstance(item, str) for item in risks):
                    prediction.risks = risks
                if isinstance(commentary, str) and commentary.strip():
                    prediction.commentary = commentary
            self._attach_simulation_summary(prediction)
            return prediction

        payload = {
            "race_data": race_data.model_dump(),
            "lessons": [lesson.model_dump() for lesson in relevant_lessons],
        }
        response = self.llm_client.generate_analysis(self.prompt_template, payload)
        return Prediction.model_validate(response)
