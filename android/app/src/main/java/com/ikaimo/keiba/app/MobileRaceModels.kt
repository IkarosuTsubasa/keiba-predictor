package com.ikaimo.keiba.app

data class MobileRaceListPayload(
    val targetDate: String,
    val targetDateLabel: String,
    val fallbackNotice: String,
    val items: List<MobileRaceItem>,
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
    val llmCards: List<MobileLlmCard>,
    val detailPath: String,
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

data class MobileLlmCard(
    val engine: String,
    val label: String,
    val decisionText: String,
    val marksText: String,
    val betSummary: String,
    val resultText: String,
    val roiText: String,
    val hit: Boolean,
    val statusLabel: String,
)
