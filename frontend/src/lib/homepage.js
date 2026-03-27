const HOME_HERO_COPY = {
  eyebrow: "独自モデル × 複数LLM比較 × レース検証",
  title: "定量モデルとLLM比較でレースを検証する競馬分析サイト",
  description:
    "いかいもAI競馬では、独自の定量モデルで各レースの有力馬と確率の偏りを抽出し、さらに複数のLLMで判断差を比較・公開しています。予想を並べるだけでなく、結果・回収率・レース後の振り返りまで継続して検証します。",
};

export const METHOD_SUMMARY_STEPS = [
  {
    step: "Step 1",
    title: "独自定量モデルで有力馬を抽出",
    description:
      "過去成績、条件適性、想定オッズ、位置取りのバランスから、各レースの有力候補と評価順の輪郭を整理します。",
  },
  {
    step: "Step 2",
    title: "LLMで買い目構成を比較",
    description:
      "同じ定量評価を受け取りながらも、複数のLLMが券種、点数、見送り判断の違いをどう出すかを並べて確認します。",
  },
  {
    step: "Step 3",
    title: "結果と履歴で継続検証",
    description:
      "公開した予想は結果とともに追跡し、単日だけでなく履歴分析までつなげてモデル差と再現性を見ていきます。",
  },
];

export const FEATURED_CONTENT_ITEMS = [
  {
    id: "methodology",
    title: "定量モデルとLLMの役割分担を読み解く",
    category: "分析方法",
    excerpt:
      "定量モデルがどこまで土台をつくり、LLM比較がどの局面で効いてくるのかを、公開フローに沿って整理した解説です。",
    tags: ["分析方法", "モデル比較", "公開フロー"],
    href: "/keiba/methodology",
  },
  {
    id: "guide",
    title: "初めてでも迷わないトップページの読み方",
    category: "ガイド",
    excerpt:
      "見どころ、深掘り分析、公開レースのどこから読めばよいかをまとめた導入ガイドです。",
    tags: ["ガイド", "導入", "比較の見方"],
    href: "/keiba/guide",
  },
  {
    id: "history",
    title: "履歴分析でモデル差と再現性を確かめる",
    category: "検証",
    excerpt:
      "単日の当たり外れではなく、期間で見たときにどのモデルがどんな局面で強いかを確認するための入口です。",
    tags: ["履歴分析", "再現性", "検証"],
    href: "/keiba/history",
  },
];

export const BEGINNER_GUIDE_LINKS = [
  {
    title: "このサイトについて",
    note: "公開ビューの役割と全体像を確認",
    href: "/keiba/about",
  },
  {
    title: "読み方ガイド",
    note: "見どころ、比較、履歴分析の読み方",
    href: "/keiba/guide",
  },
  {
    title: "分析方針",
    note: "定量モデルとLLMの役割分担",
    href: "/keiba/methodology",
  },
  {
    title: "履歴分析",
    note: "月間・年間・累計の比較を見る",
    href: "/keiba/history",
  },
];

function safeText(value) {
  return String(value || "").trim();
}

function toNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function getTokyoTodayKey() {
  const formatter = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Tokyo",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const parts = formatter.formatToParts(new Date());
  const year = parts.find((item) => item.type === "year")?.value || "";
  const month = parts.find((item) => item.type === "month")?.value || "";
  const day = parts.find((item) => item.type === "day")?.value || "";
  return year && month && day ? `${year}-${month}-${day}` : "";
}

function parseDateText(dateText) {
  const matched = safeText(dateText).match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!matched) {
    return null;
  }
  return {
    year: Number(matched[1]),
    month: Number(matched[2]),
    day: Number(matched[3]),
  };
}

function formatExplicitDate(dateText) {
  const parsed = parseDateText(dateText);
  if (!parsed) {
    return "";
  }
  return `${parsed.year}年${parsed.month}月${parsed.day}日`;
}

export function buildTargetDateContext(data) {
  const targetDate = safeText(data?.target_date);
  const targetDateLabel = safeText(data?.target_date_label) || formatExplicitDate(targetDate) || "対象日";
  const isToday = Boolean(targetDate) && targetDate === getTokyoTodayKey();
  const baseHeading = isToday ? "本日" : targetDateLabel || "対象日";

  return {
    targetDate,
    targetDateLabel,
    isToday,
    summaryHeading: `${baseHeading}のサマリー`,
    guideHeading: `${baseHeading}の見どころ`,
    raceBoardTitle: `${baseHeading}の公開レース`,
    dailyRoiTitle: `${baseHeading} ROI`,
    listCtaLabel: "レース一覧を見る",
  };
}

function parseMainHorse(marksText) {
  const matched = safeText(marksText).match(/◎\s*([0-9]+)/);
  return matched ? matched[1] : "";
}

function buildRaceHref(race, search = "") {
  const runId = encodeURIComponent(safeText(race?.run_id));
  if (!runId) {
    return search ? `/keiba?${search}` : "/keiba";
  }
  return search ? `/keiba/race/${runId}?${search}` : `/keiba/race/${runId}`;
}

function buildRaceListHref(targetDate) {
  const dateText = safeText(targetDate);
  return dateText ? `/keiba?date=${encodeURIComponent(dateText)}` : "/keiba";
}

function pickBadges(race) {
  return Array.isArray(race?.display_header?.badges)
    ? race.display_header.badges.filter(Boolean).slice(0, 3)
    : [];
}

function buildBadgeText(race, fallback = "対象レース") {
  const badges = pickBadges(race);
  return badges.length ? badges.join(" / ") : fallback;
}

function buildAgreementStats(race) {
  const cards = Array.isArray(race?.cards) ? race.cards : [];
  const validMainHorses = cards.map((card) => parseMainHorse(card?.marks_text)).filter(Boolean);
  const counts = new Map();

  for (const horseNo of validMainHorses) {
    counts.set(horseNo, (counts.get(horseNo) || 0) + 1);
  }

  let topHorse = "";
  let topCount = 0;
  for (const [horseNo, count] of counts.entries()) {
    if (count > topCount) {
      topHorse = horseNo;
      topCount = count;
    }
  }

  const noBetCount = cards.filter(
    (card) => safeText(card?.decision_text).toLowerCase() === "no_bet",
  ).length;

  return {
    cardCount: cards.length,
    validMainCount: validMainHorses.length,
    uniqueMainCount: counts.size,
    topHorse,
    topCount,
    noBetCount,
    agreementRatio: validMainHorses.length ? topCount / validMainHorses.length : 0,
  };
}

function buildAnalysisTags(stats) {
  const tags = [];

  if (stats.agreementRatio >= 0.75) {
    tags.push("高一致");
  } else if (stats.uniqueMainCount >= 3 || stats.noBetCount >= 2) {
    tags.push("分歧あり");
  }

  if (stats.topCount >= 2 && stats.agreementRatio >= 0.5) {
    tags.push("軸向き");
  }

  if (stats.uniqueMainCount >= 3 || stats.noBetCount >= 1) {
    tags.push("波乱注意");
  }

  if (!tags.length) {
    tags.push("軸向き");
  }

  return [...new Set(tags)].slice(0, 3);
}

function buildRecommendationReason(stats, leadRaceMeta, leaderLabel) {
  if (stats.agreementRatio >= 0.75) {
    return `${leaderLabel}を含む複数モデルの視点が近く、読み筋の起点として確認しやすいレースです。`;
  }
  if (stats.uniqueMainCount >= 3 || stats.noBetCount >= 2) {
    return "本命候補や見送り判断が割れやすく、複数LLMの差を比較する価値が高いレースです。";
  }
  if (stats.topCount >= 2) {
    return "軸候補の重なりが見えやすく、買い目構成の差だけを落ち着いて比較しやすいレースです。";
  }
  return `${leadRaceMeta}の条件で判断材料が揃っており、定量モデルとLLMの差を読み始める基点に向いています。`;
}

function buildLeadText(data, targetDateContext) {
  const leadRace = data?.hero?.lead_race || null;
  const raceTitle = safeText(leadRace?.race_title) || `${targetDateContext.targetDateLabel}の注目レース`;
  const raceCount = toNumber(data?.totals?.race_count, 0);
  const leaderLabel = safeText(data?.hero?.leader?.label) || "主軸モデル";
  return `${raceTitle}を起点に、対象日の${raceCount}レースを比較できます。${leaderLabel}を含む各モデルの判断差と買い目構成の違いを、同じ導線で確認できます。`;
}

export function buildHomeHeroSummary(data, search = "") {
  const targetDateContext = buildTargetDateContext(data);
  const leadRace = data?.hero?.lead_race || null;
  const raceCount = toNumber(data?.totals?.race_count, 0);
  const updatedAt = safeText(data?.generated_at_label) || "-";
  const leadRaceTitle = safeText(leadRace?.race_title) || `${targetDateContext.targetDateLabel}の注目レース`;
  const leadRaceHref = buildRaceHref(leadRace, search);
  const leadRaceMeta = buildBadgeText(leadRace, targetDateContext.targetDateLabel);
  const leadRaceStats = buildAgreementStats(leadRace);
  const leaderLabel = safeText(data?.hero?.leader?.label) || "主軸モデル";
  const settledCount = toNumber(data?.totals?.settled_count, 0);

  return {
    ...HOME_HERO_COPY,
    ...targetDateContext,
    primaryCtaLabel: targetDateContext.listCtaLabel,
    primaryCtaHref: buildRaceListHref(targetDateContext.targetDate),
    leadRaceTitle,
    leadRaceHref,
    leadRaceMeta,
    leadText: buildLeadText(data, targetDateContext),
    reasonText: buildRecommendationReason(leadRaceStats, leadRaceMeta, leaderLabel),
    analysisTags: buildAnalysisTags(leadRaceStats),
    metrics: [
      { label: "対象日", value: targetDateContext.targetDateLabel },
      { label: "データ更新", value: updatedAt },
      { label: "対象レース数", value: `${raceCount}レース` },
      { label: "結果確定", value: `${settledCount}レース` },
    ],
  };
}

function createHighlight(label, race, summary, search, agreementLevel = "") {
  return {
    label,
    title: safeText(race?.race_title) || "対象レース",
    href: buildRaceHref(race, search),
    meta: buildBadgeText(race),
    summary,
    agreementLevel,
  };
}

function pickFeaturedRace(data) {
  return data?.hero?.lead_race || (Array.isArray(data?.races) ? data.races[0] : null) || null;
}

function pickDivergenceRace(races, featuredRace) {
  const featuredRunId = safeText(featuredRace?.run_id);
  const ranked = [...(races || [])]
    .filter((race) => safeText(race?.run_id) !== featuredRunId)
    .map((race) => ({ race, stats: buildAgreementStats(race) }))
    .filter((item) => item.stats.cardCount > 0)
    .sort((left, right) => {
      if (right.stats.uniqueMainCount !== left.stats.uniqueMainCount) {
        return right.stats.uniqueMainCount - left.stats.uniqueMainCount;
      }
      if (right.stats.noBetCount !== left.stats.noBetCount) {
        return right.stats.noBetCount - left.stats.noBetCount;
      }
      return left.stats.agreementRatio - right.stats.agreementRatio;
    });
  return ranked[0]?.race || featuredRace || null;
}

function pickSolidRace(races, featuredRace, divergenceRace) {
  const excluded = new Set([safeText(featuredRace?.run_id), safeText(divergenceRace?.run_id)]);
  const ranked = [...(races || [])]
    .filter((race) => !excluded.has(safeText(race?.run_id)))
    .map((race) => ({ race, stats: buildAgreementStats(race) }))
    .filter((item) => item.stats.validMainCount > 0)
    .sort((left, right) => {
      if (right.stats.agreementRatio !== left.stats.agreementRatio) {
        return right.stats.agreementRatio - left.stats.agreementRatio;
      }
      if (right.stats.topCount !== left.stats.topCount) {
        return right.stats.topCount - left.stats.topCount;
      }
      return left.stats.uniqueMainCount - right.stats.uniqueMainCount;
    });
  return ranked[0]?.race || featuredRace || divergenceRace || null;
}

export function buildEditorialGuide(data, search = "") {
  const targetDateContext = buildTargetDateContext(data);
  const races = Array.isArray(data?.races) ? data.races : [];
  const featuredRace = pickFeaturedRace(data);
  const divergenceRace = pickDivergenceRace(races, featuredRace);
  const solidRace = pickSolidRace(races, featuredRace, divergenceRace);

  const featuredStats = buildAgreementStats(featuredRace);
  const divergenceStats = buildAgreementStats(divergenceRace);
  const solidStats = buildAgreementStats(solidRace);

  const featuredSummary = featuredStats.topHorse
    ? `◎${featuredStats.topHorse}を軸候補として見やすい一戦です。${buildBadgeText(featuredRace)}の条件で、定量モデルと各LLMの差を先に確認できます。`
    : `最初の導入として確認しやすいレースです。${buildBadgeText(featuredRace)}の条件と買い目の組み立てを先に把握できます。`;

  const divergenceSummary =
    divergenceStats.uniqueMainCount > 1
      ? `本命候補が${divergenceStats.uniqueMainCount}通りに割れており、複数LLMの見立て差が出やすいレースです。`
      : "本命は近い一方で、券種や見送り判断の差に注目しやすいレースです。";

  const solidSummary = solidStats.topHorse
    ? `◎${solidStats.topHorse}に${solidStats.topCount}モデルが重なっており、軸向きの比較をしやすいレースです。`
    : "上位評価が比較的まとまっており、無理なく読み筋を追いやすいレースです。";

  const featuredTitle = safeText(featuredRace?.race_title) || `${targetDateContext.targetDateLabel}の注目レース`;
  const raceCount = toNumber(data?.totals?.race_count, races.length);
  const intro = targetDateContext.isToday
    ? `本日は${featuredTitle}を含む${raceCount}レースを確認できます。モデル評価が揃いやすいレースと、AIごとに見解が割れやすいレースを並べて見ることで、各モデルの特徴とリスク判断の差が掴みやすくなります。`
    : `${targetDateContext.targetDateLabel}は${featuredTitle}を含む${raceCount}レースを確認できます。定量モデルの軸とLLMごとの判断差がどこでズレるかを読み取る入口として使えます。`;

  return {
    ...targetDateContext,
    intro,
    highlights: [
      createHighlight("注目", featuredRace, featuredSummary, search, featuredStats.topHorse ? "高一致" : "軸向き"),
      createHighlight(
        "分歧",
        divergenceRace,
        divergenceSummary,
        search,
        divergenceStats.uniqueMainCount > 2 ? "分歧あり" : "波乱注意",
      ),
      createHighlight("軸", solidRace, solidSummary, search, solidStats.topHorse ? "軸向き" : "高一致"),
    ],
  };
}
