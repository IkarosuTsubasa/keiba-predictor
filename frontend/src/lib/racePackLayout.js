// Presentation only: expand a dense pack without changing simulation state or order.
export const PACK_ROW_HEIGHT = 60;

export function layoutRacePack(order, { width = 820, rows = width < 560 ? 8 : 6 } = {}) {
  if (!order.length) return { horses: [], rows, height: 88 + rows * PACK_ROW_HEIGHT };
  const left = 40;
  const right = width - 42;
  const separation = 70;
  const reserve = (Math.ceil(order.length / rows) - 1) * separation;
  const leaderDistance = order[0].distance;
  const spread = Math.max(60, leaderDistance - Math.min(...order.map((h) => h.distance)));
  const scale = Math.max(0, right - left - reserve) / spread;
  const lastX = Array(rows).fill(Infinity);
  let previousX = right;
  const horses = order.map((horse) => {
    const targetX = right - Math.max(0, leaderDistance - horse.distance) * scale;
    const preferred = (horse.horse_no - 1) % rows;
    let bestRow = 0;
    let bestX = -Infinity;
    for (let row = 0; row < rows; row += 1) {
      const x = Math.min(targetX, previousX, lastX[row] - separation);
      if (x > bestX + 0.001 || (Math.abs(x - bestX) <= 0.001 && Math.abs(row - preferred) < Math.abs(bestRow - preferred))) {
        bestX = x;
        bestRow = row;
      }
    }
    lastX[bestRow] = bestX;
    previousX = bestX;
    return { ...horse, x: bestX, y: 76 + bestRow * PACK_ROW_HEIGHT, row: bestRow };
  });
  return { horses, rows, height: 88 + rows * PACK_ROW_HEIGHT };
}
