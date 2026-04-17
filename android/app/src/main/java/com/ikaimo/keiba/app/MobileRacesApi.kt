package com.ikaimo.keiba.app

import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

object MobileRacesApi {
    fun fetchRaceList(baseWebUrl: String, token: String): MobileRaceListPayload {
        val endpoint = "${baseWebUrl.trimEnd('/')}/api/mobile/v1/races"
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

    private fun parsePayload(json: JSONObject): MobileRaceListPayload {
        val data = json.optJSONObject("data") ?: JSONObject()
        return MobileRaceListPayload(
            targetDate = data.optString("target_date").trim(),
            targetDateLabel = data.optString("target_date_label").trim(),
            fallbackNotice = data.optString("fallback_notice").trim(),
            featuredRace = parseFeaturedRace(data.optJSONObject("featured_race")),
            confidenceRanking = parseConfidenceRanking(data.optJSONArray("confidence_ranking")),
            items = parseRaceItems(data.optJSONArray("items")),
        )
    }

    private fun parseFeaturedRace(json: JSONObject?): MobileFeaturedRace? {
        if (json == null) return null
        val runId = json.optString("run_id").trim()
        val detailPath = json.optString("detail_path").trim()
        val summary = parseSummary(json.optJSONObject("summary"))
        if (runId.isBlank() && detailPath.isBlank() && summary.top5.isEmpty()) return null
        return MobileFeaturedRace(
            runId = runId,
            raceId = json.optString("race_id").trim(),
            raceTitle = json.optString("race_title").trim(),
            raceName = json.optString("race_name").trim(),
            scheduledOffTime = json.optString("scheduled_off_time").trim(),
            statusLabel = json.optString("status_label").trim(),
            detailPath = detailPath,
            summary = summary,
        )
    }

    private fun parseConfidenceRanking(array: JSONArray?): List<MobileConfidenceRankingItem> {
        if (array == null) return emptyList()
        val items = mutableListOf<MobileConfidenceRankingItem>()
        for (index in 0 until array.length()) {
            val row = array.optJSONObject(index) ?: continue
            items +=
                MobileConfidenceRankingItem(
                    runId = row.optString("run_id").trim(),
                    raceId = row.optString("race_id").trim(),
                    raceTitle = row.optString("race_title").trim(),
                    statusLabel = row.optString("status_label").trim(),
                    detailPath = row.optString("detail_path").trim(),
                    mainHorseNo = row.optString("main_horse_no").trim(),
                    confidenceScore = row.optDouble("confidence_score", 0.0),
                )
        }
        return items
    }

    private fun parseRaceItems(array: JSONArray?): List<MobileRaceItem> {
        if (array == null) return emptyList()
        val items = mutableListOf<MobileRaceItem>()
        for (index in 0 until array.length()) {
            val row = array.optJSONObject(index) ?: continue
            items +=
                MobileRaceItem(
                    runId = row.optString("run_id").trim(),
                    raceId = row.optString("race_id").trim(),
                    raceTitle = row.optString("race_title").trim(),
                    raceName = row.optString("race_name").trim(),
                    location = row.optString("location").trim(),
                    scheduledOffTime = row.optString("scheduled_off_time").trim(),
                    status = row.optString("status").trim(),
                    statusLabel = row.optString("status_label").trim(),
                    result = parseResult(row.optJSONObject("result")),
                    summary = parseSummary(row.optJSONObject("summary")),
                    llmCards = parseLlmCards(row.optJSONArray("llm_cards")),
                    detailPath = row.optString("detail_path").trim(),
                )
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

    private fun parseSummary(json: JSONObject?): MobileRaceSummary {
        return MobileRaceSummary(
            mainHorseNo = json?.optString("main_horse_no").orEmpty().trim(),
            confidenceScore = json?.optDouble("confidence_score", 0.0) ?: 0.0,
            agreementScore = json?.optDouble("agreement_score", 0.0) ?: 0.0,
            modelCount = json?.optInt("model_count", 0) ?: 0,
            top5 = parseSummaryTop5(json?.optJSONArray("top5")),
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

    private fun parseLlmCards(array: JSONArray?): List<MobileLlmCard> {
        if (array == null) return emptyList()
        val items = mutableListOf<MobileLlmCard>()
        for (index in 0 until array.length()) {
            val row = array.optJSONObject(index) ?: continue
            items +=
                MobileLlmCard(
                    engine = row.optString("engine").trim(),
                    label = row.optString("label").trim(),
                    decisionText = row.optString("decision_text").trim(),
                    marksText = row.optString("marks_text").trim(),
                    betSummary = row.optString("bet_summary").trim(),
                    resultText = row.optString("result_text").trim(),
                    roiText = row.optString("roi_text").trim(),
                    hit = row.optBoolean("hit"),
                    statusLabel = row.optString("status_label").trim(),
                )
        }
        return items
    }
}
