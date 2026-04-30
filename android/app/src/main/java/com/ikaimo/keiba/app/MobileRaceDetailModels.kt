package com.ikaimo.keiba.app

data class MobileRaceDetailPayload(
    val targetDate: String,
    val targetDateLabel: String,
    val fallbackNotice: String,
    val race: MobileRaceDetail?,
)

data class MobileRaceDetail(
    val runId: String,
    val raceId: String,
    val raceTitle: String,
    val raceName: String,
    val scheduledOffTime: String,
    val displayVariant: String,
    val statusLabel: String,
    val detailTitle: String,
    val badges: List<String>,
    val result: MobileRaceResult,
    val resultText: String,
    val predictorCompareCards: List<MobilePredictorCompareCard>,
    val top5: List<MobileSummaryHorse>,
    val predictorTop5: Map<String, List<MobilePredictorRankingHorse>>,
    val confidenceScore: Double,
    val agreementScore: Double,
    val conditionRanking: MobileConditionPredictorRanking,
)

data class MobilePredictorCompareCard(
    val predictorId: String,
    val label: String,
    val marksText: String,
    val isPlaceholder: Boolean,
    val placeholderText: String,
    val topHorses: List<MobilePredictorRankingHorse>,
)

data class MobilePredictorRankingHorse(
    val horseNo: String,
    val horseName: String,
    val predRank: Int,
    val top3ProbModel: Double,
    val rankScoreNorm: Double,
)

data class MobileConditionPredictorRanking(
    val conditionText: String,
    val sampleCount: Int,
    val cards: List<MobileConditionPredictorCard>,
)

data class MobileConditionPredictorCard(
    val predictorId: String,
    val label: String,
    val rank: Int,
    val top5ToTop3HitRateText: String,
)
