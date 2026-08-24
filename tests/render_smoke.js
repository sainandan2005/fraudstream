const fs = require("fs");

const html = fs.readFileSync("src/alerts_api/static/index.html", "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function stubEl() {
  return {
    innerHTML: "", textContent: "", value: "", checked: false,
    classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
    addEventListener() {},
  };
}
const stubDoc = {
  getElementById: () => stubEl(),
  querySelectorAll: () => [],
  querySelector: () => stubEl(),
  addEventListener() {},
};

let fetchCalls = 0;
const stubFetch = () => {
  fetchCalls++;
  return Promise.resolve({ ok: true, status: 200, json: async () => ({ txns_processed: 1, alerts_raised: 0, last_processed_ms: 0, rule_hits: {} }) });
};

const factory = new Function(
  "document", "fetch", "setInterval", "setTimeout", "window",
  `${script}\n;return { stateBoxes, timelineHtml };`
);
const api = factory(stubDoc, stubFetch, () => {}, () => {}, globalThis);

setTimeout(async () => {
  const payload = JSON.parse(fs.readFileSync(process.env.TEMP + "/opencode/card.json", "utf8"));

  const state = api.stateBoxes(payload);
  if (!state.includes("Txns traced") || !state.includes("Velocity")) throw new Error("stateBoxes missing fields");
  const timeline = api.timelineHtml(payload.transactions);
  if (!timeline.includes("titem") || !timeline.includes("lag")) throw new Error("timeline malformed");

  const flagged = { ...payload, transactions: payload.transactions.map(t => ({ ...t, score: Math.max(t.score, 75), rules: t.rules.length ? t.rules : [{ rule: "geo_velocity", weight: 75, detail: "test" }] })) };
  const ftl = api.timelineHtml(flagged.transactions);
  if (!ftl.includes("badge")) throw new Error("flagged badge missing");

  const empty = { card_id: "x", txn_count: 0, flagged_count: 0, transactions: [] };
  api.stateBoxes(empty); api.timelineHtml(empty);

  await new Promise(r => setTimeout(r, 50));
  if (fetchCalls === 0) throw new Error("initial refresh() never ran");

  console.log("RENDER SMOKE OK, state:", state.length, "chars | timeline:", payload.transactions.length, "items | flagged-badge:", ftl.includes("badge"), "| empty-card safe | live-loop ran");
}, 10);
