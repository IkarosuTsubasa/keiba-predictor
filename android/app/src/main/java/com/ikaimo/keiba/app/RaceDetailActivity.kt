package com.ikaimo.keiba.app

import android.content.Context
import android.content.Intent
import android.graphics.Typeface
import android.os.Bundle
import android.util.TypedValue
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import com.google.android.material.snackbar.Snackbar
import com.ikaimo.keiba.app.databinding.ActivityRaceDetailBinding

class RaceDetailActivity : AppCompatActivity() {
    private lateinit var binding: ActivityRaceDetailBinding

    private val runId: String by lazy {
        intent.getStringExtra(EXTRA_RUN_ID).orEmpty().trim()
    }
    private val targetDate: String by lazy {
        intent.getStringExtra(EXTRA_TARGET_DATE).orEmpty().trim()
    }
    private val initialTitle: String by lazy {
        intent.getStringExtra(EXTRA_TITLE).orEmpty().trim()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityRaceDetailBinding.inflate(layoutInflater)
        setContentView(binding.root)
        EdgeToEdgeUi.apply(window, binding.raceDetailRoot, binding.toolbar)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)
        binding.toolbar.setNavigationOnClickListener { finish() }
        binding.toolbarTitle.text = initialTitle.ifBlank { getString(R.string.title_race_detail) }
        binding.toolbarSubtitle.text = getString(R.string.subtitle_race_detail_native)
        binding.swipeRefresh.setOnRefreshListener {
            loadRaceDetail(forceRefresh = true)
        }

        if (runId.isBlank()) {
            showEmpty(getString(R.string.race_detail_empty))
            return
        }

        loadRaceDetail(forceRefresh = false)
    }

    private fun loadRaceDetail(forceRefresh: Boolean) {
        val token = BuildConfig.MOBILE_API_TOKEN.trim()
        if (token.isBlank()) {
            showEmpty(getString(R.string.race_detail_empty))
            showLoadError(IllegalStateException("MOBILE_API_TOKEN is blank"), forceRefresh)
            return
        }

        showLoading(true, forceRefresh)
        Thread {
            try {
                val payload = MobileRaceDetailApi.fetchRaceDetail(BuildConfig.BASE_WEB_URL, token, runId, targetDate)
                runOnUiThread {
                    showLoading(false, forceRefresh)
                    renderPayload(payload)
                }
            } catch (error: Throwable) {
                runOnUiThread {
                    showLoading(false, forceRefresh)
                    if (!binding.contentScroll.isVisible) {
                        showEmpty(getString(R.string.race_detail_empty))
                    }
                    showLoadError(error, forceRefresh)
                }
            }
        }.start()
    }

    private fun renderPayload(payload: MobileRaceDetailPayload) {
        binding.detailDateLabel.text = payload.targetDateLabel.ifBlank { payload.targetDate }
        binding.detailFallbackNotice.text = payload.fallbackNotice
        binding.detailFallbackNotice.isVisible = payload.fallbackNotice.isNotBlank()

        val race = payload.race
        if (race == null) {
            showEmpty(getString(R.string.race_detail_empty))
            return
        }

        binding.contentScroll.isVisible = true
        binding.detailEmpty.isVisible = false

        binding.toolbarTitle.text = race.raceTitle.ifBlank { getString(R.string.title_race_detail) }
        binding.toolbarSubtitle.text =
            listOf(
                payload.targetDateLabel.ifBlank { payload.targetDate },
                race.raceName.takeIf { it.isNotBlank() },
            ).filterNotNull().joinToString("  ").ifBlank { getString(R.string.subtitle_race_detail_native) }

        binding.detailTitle.text = race.detailTitle.ifBlank { race.raceTitle.ifBlank { getString(R.string.title_race_detail) } }
        binding.detailBadges.text = race.badges.joinToString(" / ")
        binding.detailBadges.isVisible = race.badges.isNotEmpty()
        binding.detailStatusValue.text = race.statusLabel.ifBlank { getString(R.string.race_detail_status_default) }
        binding.detailConfidenceValue.text = formatPercent(race.confidenceScore)
        binding.detailAgreementValue.text = formatPercent(race.agreementScore)

        renderResultSection(race)
        renderAiIndexSection(race)
        renderCompareSection(race)
        renderConditionSection(race.conditionRanking)
    }

    private fun renderResultSection(race: MobileRaceDetail) {
        binding.detailResultList.removeAllViews()
        val hasResult = race.result.isSettled && race.result.top3.isNotEmpty()
        binding.detailResultEmpty.isVisible = !hasResult
        binding.detailResultEmpty.text =
            if (hasResult) {
                ""
            } else {
                race.resultText.ifBlank { getString(R.string.race_detail_result_pending) }
            }
        if (!hasResult) return

        race.result.top3.forEachIndexed { index, item ->
            binding.detailResultList.addView(
                buildDoubleLineRow(
                    leading = "${item.rank}着",
                    title = listOf(item.horseNo.takeIf { it.isNotBlank() }, item.horseName.takeIf { it.isNotBlank() })
                        .filterNotNull()
                        .joinToString(" "),
                    trailing = "",
                    topMarginDp = if (index == 0) 0 else 10,
                ),
            )
        }
    }

    private fun renderAiIndexSection(race: MobileRaceDetail) {
        binding.detailAiIndexList.removeAllViews()
        val rows = buildAiIndexRows(race)
        binding.detailAiIndexEmpty.isVisible = rows.isEmpty()
        if (rows.isEmpty()) return

        rows.take(5).forEachIndexed { index, item ->
            binding.detailAiIndexList.addView(
                buildDoubleLineRow(
                    leading = item.horseNo,
                    title = item.horseName,
                    trailing = item.scoreText,
                    topMarginDp = if (index == 0) 0 else 10,
                ),
            )
        }
    }

    private fun renderCompareSection(race: MobileRaceDetail) {
        binding.detailCompareList.removeAllViews()
        val cards = buildCompareCards(race)
        binding.detailCompareEmpty.isVisible = cards.isEmpty()
        if (cards.isEmpty()) return

        cards.forEachIndexed { index, item ->
            binding.detailCompareList.addView(
                buildSectionRow(
                    title = item.label.ifBlank { item.predictorId.ifBlank { "-" } },
                    body = if (item.isPlaceholder) {
                        item.placeholderText.ifBlank { getString(R.string.race_detail_compare_placeholder) }
                    } else {
                        item.marksText.ifBlank { getString(R.string.race_detail_compare_placeholder) }
                    },
                    topMarginDp = if (index == 0) 0 else 10,
                ),
            )
        }
    }

    private fun renderConditionSection(ranking: MobileConditionPredictorRanking) {
        binding.detailConditionMeta.isVisible = ranking.conditionText.isNotBlank() || ranking.sampleCount > 0
        binding.detailConditionMeta.text =
            listOf(
                ranking.conditionText.takeIf { it.isNotBlank() },
                ranking.sampleCount.takeIf { it > 0 }?.let { getString(R.string.race_detail_condition_samples, it) },
            ).filterNotNull().joinToString(" / ")

        binding.detailConditionList.removeAllViews()
        binding.detailConditionEmpty.isVisible = ranking.cards.isEmpty()
        if (ranking.cards.isEmpty()) return

        ranking.cards.forEachIndexed { index, item ->
            binding.detailConditionList.addView(
                buildDoubleLineRow(
                    leading = item.rank.takeIf { it > 0 }?.toString() ?: "-",
                    title = item.label.ifBlank { "-" },
                    trailing = item.top5ToTop3HitRateText.ifBlank { "-" },
                    topMarginDp = if (index == 0) 0 else 10,
                ),
            )
        }
    }

    private fun showLoading(loading: Boolean, refresh: Boolean) {
        binding.progressIndicator.isVisible = loading
        binding.detailProgress.isVisible = loading && !refresh
        if (!loading) {
            binding.swipeRefresh.isRefreshing = false
        } else if (refresh) {
            binding.swipeRefresh.isRefreshing = true
        }
    }

    private fun showEmpty(message: String) {
        binding.contentScroll.isVisible = false
        binding.detailEmpty.isVisible = true
        binding.detailEmpty.text = message
        binding.swipeRefresh.isRefreshing = false
        binding.progressIndicator.isVisible = false
        binding.detailProgress.isVisible = false
    }

    private fun showLoadError(error: Throwable, refresh: Boolean) {
        val message =
            if (BuildConfig.DEBUG) {
                getString(
                    R.string.race_detail_load_failed_debug,
                    getString(R.string.race_detail_load_failed),
                    error.message ?: error.javaClass.simpleName,
                )
            } else {
                getString(R.string.race_detail_load_failed)
            }
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
            .setAction(R.string.retry) {
                loadRaceDetail(forceRefresh = true)
            }
            .show()
        if (!refresh) {
            binding.swipeRefresh.isRefreshing = false
        }
    }

    private fun buildAiIndexRows(race: MobileRaceDetail): List<AiIndexRow> {
        if (race.top5.isNotEmpty()) {
            return race.top5.map { item ->
                AiIndexRow(
                    horseNo = item.horseNo.ifBlank { "-" },
                    horseName = item.horseName.ifBlank { "-" },
                    scoreText = item.supportScore.toString(),
                )
            }
        }

        val rows = linkedMapOf<String, AiIndexAccumulator>()
        val markWeight = mapOf("◎" to 5, "○" to 4, "▲" to 3, "△" to 2, "☆" to 1)
        val pattern = Regex("([◎○▲△☆])\\s*([0-9]+)")
        race.predictorCompareCards.filter { !it.isPlaceholder }.forEach { card ->
            val horseNameMap = card.topHorses.associateBy({ it.horseNo }, { it.horseName })
            pattern.findAll(card.marksText).forEach { match ->
                val symbol = match.groupValues.getOrNull(1).orEmpty()
                val horseNo = match.groupValues.getOrNull(2).orEmpty().trim()
                if (horseNo.isBlank()) return@forEach
                val current = rows.getOrPut(horseNo) { AiIndexAccumulator(horseNo = horseNo) }
                current.score += markWeight[symbol] ?: 0
                if (current.horseName.isBlank()) {
                    current.horseName = horseNameMap[horseNo].orEmpty()
                }
            }
        }

        if (rows.isNotEmpty()) {
            return rows.values
                .sortedWith(compareByDescending<AiIndexAccumulator> { it.score }.thenBy { it.horseNo.toIntOrNull() ?: 999 })
                .map { item ->
                    AiIndexRow(
                        horseNo = item.horseNo,
                        horseName = item.horseName.ifBlank { "-" },
                        scoreText = item.score.toString(),
                    )
                }
        }

        val fallback = mutableListOf<AiIndexRow>()
        for (predictorId in listOf("main", "v6_kiwami")) {
            for (item in race.predictorTop5[predictorId].orEmpty()) {
                if (fallback.any { it.horseNo == item.horseNo }) continue
                fallback +=
                    AiIndexRow(
                        horseNo = item.horseNo.ifBlank { "-" },
                        horseName = item.horseName.ifBlank { "-" },
                        scoreText = "40",
                    )
                if (fallback.size >= 5) return fallback
            }
        }
        return fallback
    }

    private fun buildCompareCards(race: MobileRaceDetail): List<MobilePredictorCompareCard> {
        val order = listOf("main", "v2_opus", "v3_premium", "v4_gemini", "v5_stacking", "v6_kiwami")
        val labels =
            mapOf(
                "main" to "ゲート",
                "v2_opus" to "ストライド",
                "v3_premium" to "伯楽",
                "v4_gemini" to "馬場眼",
                "v5_stacking" to "フュージョン",
                "v6_kiwami" to "極 KIWAMI",
            )
        val primary = race.predictorCompareCards.associateBy { it.predictorId.ifBlank { it.label } }
        val marks = listOf("◎", "○", "▲", "△", "☆")
        val items = mutableListOf<MobilePredictorCompareCard>()

        for (predictorId in order) {
            val existing = primary[predictorId]
            if (existing != null) {
                items += existing
                continue
            }
            val ranking = race.predictorTop5[predictorId].orEmpty()
            if (ranking.isNotEmpty()) {
                val marksText =
                    ranking.take(marks.size).mapIndexedNotNull { index, item ->
                        val horseNo = item.horseNo.ifBlank { return@mapIndexedNotNull null }
                        "${marks[index]}$horseNo"
                    }.joinToString(" ")
                items +=
                    MobilePredictorCompareCard(
                        predictorId = predictorId,
                        label = labels[predictorId].orEmpty().ifBlank { predictorId },
                        marksText = marksText,
                        isPlaceholder = false,
                        placeholderText = "",
                        topHorses = ranking,
                    )
            }
        }

        return if (items.isNotEmpty()) items else race.predictorCompareCards
    }

    private fun buildSectionRow(title: String, body: String, topMarginDp: Int): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams =
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                ).apply {
                    topMargin = dp(topMarginDp)
                }
            setPadding(dp(12), dp(12), dp(12), dp(12))
            background = ContextCompat.getDrawable(context, R.drawable.bg_native_race_status)

            addView(
                TextView(context).apply {
                    text = title
                    setTextColor(ContextCompat.getColor(context, R.color.text_primary))
                    setTypeface(typeface, Typeface.BOLD)
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
                },
            )
            addView(
                TextView(context).apply {
                    text = body
                    setTextColor(ContextCompat.getColor(context, R.color.text_secondary))
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
                    setPadding(0, dp(6), 0, 0)
                },
            )
        }
    }

    private fun buildDoubleLineRow(leading: String, title: String, trailing: String, topMarginDp: Int): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams =
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                ).apply {
                    topMargin = dp(topMarginDp)
                }
            gravity = android.view.Gravity.CENTER_VERTICAL

            addView(
                TextView(context).apply {
                    text = leading
                    setTextColor(ContextCompat.getColor(context, R.color.text_secondary))
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
                    layoutParams =
                        LinearLayout.LayoutParams(dp(32), LinearLayout.LayoutParams.WRAP_CONTENT)
                },
            )

            addView(
                TextView(context).apply {
                    text = title
                    setTextColor(ContextCompat.getColor(context, R.color.text_primary))
                    setTypeface(typeface, Typeface.BOLD)
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
                    layoutParams =
                        LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                },
            )

            addView(
                TextView(context).apply {
                    text = trailing
                    setTextColor(ContextCompat.getColor(context, R.color.accent_color_dark))
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
                },
            )
        }
    }

    private fun dp(value: Int): Int {
        return (value * resources.displayMetrics.density).toInt()
    }

    private fun formatPercent(value: Double): String {
        return "${(value.coerceIn(0.0, 1.0) * 100.0).toInt()}%"
    }

    private data class AiIndexAccumulator(
        val horseNo: String,
        var horseName: String = "",
        var score: Int = 0,
    )

    private data class AiIndexRow(
        val horseNo: String,
        val horseName: String,
        val scoreText: String,
    )

    companion object {
        private const val EXTRA_RUN_ID = "extra_run_id"
        private const val EXTRA_TARGET_DATE = "extra_target_date"
        private const val EXTRA_TITLE = "extra_title"

        fun createIntent(context: Context, runId: String, targetDate: String = "", title: String = ""): Intent {
            return Intent(context, RaceDetailActivity::class.java).apply {
                putExtra(EXTRA_RUN_ID, runId)
                putExtra(EXTRA_TARGET_DATE, targetDate)
                putExtra(EXTRA_TITLE, title)
            }
        }
    }
}
