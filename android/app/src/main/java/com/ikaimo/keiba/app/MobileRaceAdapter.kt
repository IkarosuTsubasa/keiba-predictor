package com.ikaimo.keiba.app

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.ikaimo.keiba.app.databinding.ItemNativeRaceBinding

class MobileRaceAdapter(
    private val onRaceSelected: (MobileRaceItem) -> Unit,
) : RecyclerView.Adapter<MobileRaceAdapter.RaceViewHolder>() {
    private val items = mutableListOf<MobileRaceItem>()

    fun submitList(nextItems: List<MobileRaceItem>) {
        items.clear()
        items.addAll(nextItems)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RaceViewHolder {
        val binding =
            ItemNativeRaceBinding.inflate(
                LayoutInflater.from(parent.context),
                parent,
                false,
            )
        return RaceViewHolder(binding, onRaceSelected)
    }

    override fun onBindViewHolder(holder: RaceViewHolder, position: Int) {
        holder.bind(items[position])
    }

    override fun getItemCount(): Int = items.size

    class RaceViewHolder(
        private val binding: ItemNativeRaceBinding,
        private val onRaceSelected: (MobileRaceItem) -> Unit,
    ) : RecyclerView.ViewHolder(binding.root) {
        fun bind(item: MobileRaceItem) {
            val detailAvailable = item.detailPath.isNotBlank()
            binding.raceTitle.text = "${item.location}${item.raceId}"
            binding.raceSubtitle.text =
                listOf(item.raceName.takeIf { it.isNotBlank() }, item.scheduledOffTime.takeIf { it.isNotBlank() })
                    .filterNotNull()
                    .joinToString("  ")
            binding.raceStatus.text = item.statusLabel
            binding.raceResult.text = buildResultText(item.result)
            binding.raceLlmSummary.text = buildLlmSummary(item.llmCards)
            binding.root.isEnabled = detailAvailable
            binding.root.isClickable = detailAvailable
            binding.root.alpha = if (detailAvailable) 1f else 0.72f
            binding.root.setOnClickListener(if (detailAvailable) { { onRaceSelected(item) } } else null)
        }

        private fun buildResultText(result: MobileRaceResult): String {
            if (!result.isSettled || result.top3.isEmpty()) {
                return binding.root.context.getString(R.string.races_result_pending)
            }
            return result.top3.joinToString("\n") { item ->
                "${item.rank}着 ${item.horseNo} ${item.horseName}".trim()
            }
        }

        private fun buildLlmSummary(cards: List<MobileLlmCard>): String {
            if (cards.isEmpty()) {
                return binding.root.context.getString(R.string.races_llm_summary_empty)
            }
            return cards.joinToString("\n\n") { card ->
                listOf(
                    card.label,
                    card.marksText.takeIf { it.isNotBlank() }?.let { "印 $it" },
                    card.betSummary.takeIf { it.isNotBlank() }?.let { "買い目 $it" },
                    card.roiText.takeIf { it.isNotBlank() }?.let { "回収率 $it" },
                ).filterNotNull().joinToString("\n")
            }
        }
    }
}
