import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

// Compile the actual JSX without a second React runtime or generated test files.
const bundle = await build({
  stdin: {
    contents: `import React from "react";
      import {renderToStaticMarkup} from "react-dom/server";
      import Preview from "../components/RaceSimulatorPreview.jsx";
      export const render = props => renderToStaticMarkup(React.createElement(Preview, props));`,
    resolveDir: fileURLToPath(new URL(".", import.meta.url)),
    loader: "jsx",
  },
  bundle: true, write: false, platform: "node", format: "cjs",
  loader: { ".css": "empty" }, logLevel: "silent",
});
const compiled = { exports: {} };
new Function("require", "module", "exports", bundle.outputFiles[0].text)(createRequire(import.meta.url), compiled, compiled.exports);

test("详情日期尚未到达时保留占位节点，避免监听空节点导致整页崩溃", () => {
  const markup = compiled.exports.render({ context: { raceId: "202645061908", date: "" } });
  assert.match(markup, /展開シミュレーション/);
  assert.match(markup, /データ準備中/);
  assert.match(markup, /<section/);
});

test("详情接口提供日期后渲染等待状态，不要求比赛对象重复提供日期", () => {
  const detailData = { target_date: "2026-06-19", race: { race_id: "202645061908" } };
  const markup = compiled.exports.render({ context: { raceId: detailData.race.race_id, date: detailData.target_date } });
  assert.match(markup, /データを準備しています/);
  assert.match(markup, /<section/);
});
