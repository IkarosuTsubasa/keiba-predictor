package com.ikaimo.keiba.app

data class MobileRaceListPayload(
    val targetDate: String,
    val targetDateLabel: String,
    val fallbackNotice: String,
    val featuredRace: MobileFeaturedRace?,
    val confidenceRanking: List<MobileConfidenceRankingItem>,
    val items: List<MobileRaceItem>,
)

data class MobileFeaturedRace(
    val runId: String,
    val raceId: String,
    val raceTitle: String,
    val raceName: String,
    val scheduledOffTime: String,
    val statusLabel: String,
    val detailPath: String,
    val summary: MobileRaceSummary,
)

data class MobileConfidenceRankingItem(
    val runId: String,
    val raceId: String,
    val raceTitle: String,
    val statusLabel: String,
    val detailPath: String,
    val mainHorseNo: String,
    val confidenceScore: Double,
)

data class MobileRaceItem(
    val runId: String,
    val raceId: String,
    val raceTitle: String,
    val raceName: String,
    val location: String,
    val scheduledOffTime: String,
    val status: String,
    val statusLabel: String,
    val result: MobileRaceResult,
    val summary: MobileRaceSummary,
    val detailPath: String,
)

data class MobileRaceSummary(
    val mainHorseNo: String,
    val confidenceScore: Double,
    val agreementScore: Double,
    val modelCount: Int,
    val top5: List<MobileSummaryHorse>,
)

data class MobileSummaryHorse(
    val horseNo: String,
    val horseName: String,
    val supportScore: Int,
)

data class MobileRaceResult(
    val isSettled: Boolean,
    val top3: List<MobileRaceFinish>,
)

data class MobileRaceFinish(
    val rank: Int,
    val horseNo: String,
    val horseName: String,
)
