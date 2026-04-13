import React, { useEffect } from "react";
import { getEzoicSlotId, requestEzoicAds } from "../lib/ezoic";

export default function EzoicAdSlot({ slot, wrapperClassName = "" }) {
  const slotId = getEzoicSlotId(slot);

  useEffect(() => {
    if (!slotId) {
      return;
    }
    requestEzoicAds([slotId]);
  }, [slotId]);

  if (!slotId) {
    return null;
  }

  const className = ["ezoic-ad-slot", wrapperClassName].filter(Boolean).join(" ");
  return (
    <div className={className}>
      <div id={slotId} />
    </div>
  );
}
