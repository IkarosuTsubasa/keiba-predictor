import React from "react";
import { RACE_DETAIL_AFFILIATE } from "../content/affiliateContent";

export default function RaceDetailAffiliateCard() {
  if (!RACE_DETAIL_AFFILIATE?.href || !RACE_DETAIL_AFFILIATE?.imageSrc) {
    return null;
  }

  return (
    <aside className="race-detail-affiliate-card" aria-label="Amazon affiliate">
      <a
        className="race-detail-affiliate-card__link"
        href={RACE_DETAIL_AFFILIATE.href}
        target="_blank"
        rel="noopener noreferrer sponsored"
        aria-label={RACE_DETAIL_AFFILIATE.ariaLabel || RACE_DETAIL_AFFILIATE.alt}
      >
        <img
          className="race-detail-affiliate-card__image"
          src={RACE_DETAIL_AFFILIATE.imageSrc}
          alt={RACE_DETAIL_AFFILIATE.alt}
          loading="lazy"
        />
      </a>
    </aside>
  );
}
