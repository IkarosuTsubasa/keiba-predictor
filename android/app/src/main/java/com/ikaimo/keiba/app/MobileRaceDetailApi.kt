package com.ikaimo.keiba.app

import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.nio.charset.StandardCharsets

object MobileRaceDetailApi {
    fun fetchRaceDetail(baseWebUrl: String, token: String, runId: String, targetDate: String = ""): MobileRaceDetailPayload {
        val encodedRunId = URLEncoder.encode(runId.trim(), StandardCharsets.UTF_8.name())
        val dateQuery =
            targetDate.trim().takeIf { it.isNotBlank() }?.let {
                "?date=${URLEncoder.encode(it, StandardCharsets.UTF_8.name())}"
            }.orEmpty()
        val endpoint = "${baseWebUrl.trimEnd('/')}/api/mobile/v1/races/$encodedRunId$dateQuery"
        val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 10_000
            readTimeout = 15_000
            setRequestProperty("Accept", "application/json")
            setRequestProperty("X-App-Client", "android")
            setRequestProperty("X-App-Token", token)
        }

        try {
            val statusCode = connection.responseCode
            val stream =
                if (statusCode in 200..299) {
                    connection.inputStream
                } else {
                    connection.errorStream ?: connection.inputStream
                }
            val body =
                BufferedReader(InputStreamReader(stream, StandardCharsets.UTF_8)).use { reader ->
                    buildString {
                        while (true) {
                            val line = reader.readLine() ?: break
                            append(line)
                        }
                    }
                }

            if (statusCode !in 200..299) {
                throw IllegalStateException("HTTP $statusCode: $body")
            }

            return parsePayload(JSONObject(body))
        } finally {
            connection.disconnect()
        }
    }

    private fun parsePayload(json: JSONObject): MobileRaceDetailPayload {
        val data = json.optJSONObject("data") ?: JSONObject()
        return MobileRaceDetailPayload(
            targetDate = data.optString("target_date").trim(),
            targetDateLabel = data.optString("target_date_label").trim(),
            fallbackNotice = data.optString("fallback_notice").trim(),
            race = parseRace(data.optJSONObject("race")),
        )
    }

    private fun parseRace(json: JSONObject?): MobileRaceDetail? {
        if (json == null) return null
        val runId = json.optString("run_id").trim()
        val detailTitle =
            json.optJSONObject("display_header")?.optString("detail_title").orEmpty().trim()
                .ifBlank { json.optJSONObject("display_header")?.optString("title").orEmpty().trim() }
        val compareCards = parsePredictorCompareCards(json.optJSONArray("predictor_compare_cards"))
        val top5 = parseSummaryTop5(json.optJSONArray("top5"))
        if (runId.isBlank() && detailTitle.isBlank() && compareCards.isEmpty() && top5.isEmpty()) return null
        return MobileRaceDetail(
            runId = runId,
            raceId = json.optString("race_id").trim(),
            raceTitle = json.optString("race_title").trim(),
            raceName = json.optString("race_name").trim(),
            scheduledOffTime = json.optString("scheduled_off_time").trim(),
            displayVariant = json.optString("display_variant").trim(),
            statusLabel = json.optJSONObject("display_status")?.optString("label").orEmpty().trim(),
            detailTitle = detailTitle,
            badges = parseStringArray(json.optJSONObject("display_header")?.optJSONArray("badges")),
            result = parseResult(json.optJSONObject("actual_result")),
            resultText = json.optJSONObject("display_body")?.optString("result_text").orEmpty().trim(),
            predictorCompareCards = compareCards,
            top5 = top5,
            predictorTop5 = parsePredictorTop5(json.optJSONObject("predictor_top5")),
            confidenceScore = json.optDouble("confidence_score", 0.0),
            agreementScore = json.optDouble("agreement_score", 0.0),
            conditionRanking = parseConditionRanking(json.optJSONObject("condition_predictor_ranking")),
        )
    }

    private fun parsePredictorCompareCards(array: JSONArray?): List<MobilePredictorCompareCard> {
        if (array == null) return emptyList()
        val items = mutableListOf<MobilePredictorCompareCard>()
        for (index in 0 until array.length()) {
            val row = array.optJSONObject(index) ?: continue
            items +=
                MobilePredictorCompareCard(
                    predictorId = row.optString("predictor_id").trim(),
                    label = row.optString("label").trim(),
                    marksText = row.optString("marks_text").trim(),
                    isPlaceholder = row.optBoolean("is_placeholder"),
                    placeholderText = row.optString("placeholder_text").trim(),
                    topHorses = parseRankingHorses(row.optJSONArray("top_horses")),
                )
        }
        return items
    }

    private fun parseRankingHorses(array: JSONArray?): List<MobilePredictorRankingHorse> {
        if (array == null) return emptyList()
        val items = mutableListOf<MobilePredictorRankingHorse>()
        for (index in 0 until array.length()) {
            val row = array.optJSONObject(index) ?: continue
            items +=
                MobilePredictorRankingHorse(
                    horseNo = row.optString("horse_no").trim(),
                    horseName = row.optString("horse_name").trim(),
                    predRank = row.optInt("pred_rank", index + 1),
                    top3ProbModel = row.optDouble("top3_prob_model", 0.0),
                    rankScoreNorm = row.optDouble("rank_score_norm", 0.0),
                )
        }
        return items
    }

    private fun parsePredictorTop5(json: JSONObject?): Map<String, List<MobilePredictorRankingHorse>> {
        if (json == null) return emptyMap()
        val items = linkedMapOf<String, List<MobilePredictorRankingHorse>>()
        val keys = json.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            items[key] = parseRankingHorses(json.optJSONArray(key))
        }
        return items
    }

    private fun parseConditionRanking(json: JSONObject?): MobileConditionPredictorRanking {
        val cards = mutableListOf<MobileConditionPredictorCard>()
        val array = json?.optJSONArray("cards")
        if (array != null) {
            for (index in 0 until array.length()) {
                val row = array.optJSONObject(index) ?: continue
                cards +=
                    MobileConditionPredictorCard(
                        predictorId = row.optString("predictor_id").trim(),
                        label = row.optString("label").trim(),
                        rank = row.optInt("rank", index + 1),
                        top5ToTop3HitRateText = row.optString("top5_to_top3_hit_rate_text").trim(),
                    )
            }
        }
        return MobileConditionPredictorRanking(
            conditionText = json?.optString("condition_text").orEmpty().trim(),
            sampleCount = json?.optInt("sample_count", 0) ?: 0,
            cards = cards,
        )
    }

    private fun parseStringArray(array: JSONArray?): List<String> {
        if (array == null) return emptyList()
        val items = mutableListOf<String>()
        for (index in 0 until array.length()) {
            val value = array.optString(index).trim()
            if (value.isNotBlank()) items += value
        }
        return items
    }

    private fun parseResult(json: JSONObject?): MobileRaceResult {
        val top3 = mutableListOf<MobileRaceFinish>()
        val array = json?.optJSONArray("top3")
        if (array != null) {
            for (index in 0 until array.length()) {
                val row = array.optJSONObject(index) ?: continue
                top3 +=
                    MobileRaceFinish(
                        rank = row.optInt("rank", index + 1),
                        horseNo = row.optString("horse_no").trim(),
                        horseName = row.optString("horse_name").trim(),
                    )
            }
        }
        return MobileRaceResult(
            isSettled = json?.optBoolean("is_settled") == true,
            top3 = top3,
        )
    }

    private fun parseSummaryTop5(array: JSONArray?): List<MobileSummaryHorse> {
        if (array == null) return emptyList()
        val items = mutableListOf<MobileSummaryHorse>()
        for (index in 0 until array.length()) {
            val row = array.optJSONObject(index) ?: continue
            items +=
                MobileSummaryHorse(
                    horseNo = row.optString("horse_no").trim(),
                    horseName = row.optString("horse_name").trim(),
                    supportScore = row.optInt("support_score", 0),
                )
        }
        return items
    }
}
