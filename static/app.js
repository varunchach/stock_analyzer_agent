const form = document.getElementById("query-form");
const input = document.getElementById("query-input");
const submitBtn = document.getElementById("submit-btn");
const chatMessages = document.getElementById("chat-messages");
const emptyState = document.getElementById("empty-state");
const newChatBtn = document.getElementById("new-chat-btn");

const TOOL_LABELS = {
  resolve_ticker: "Resolving ticker symbol",
  get_live_quote: "Fetching live price",
  get_fundamental_snapshot: "Pulling fundamentals",
  get_technical_snapshot: "Computing technicals (RSI/SMA)",
  get_price_history: "Loading price history",
  search_stock_news: "Searching latest news",
  search_market_sentiment: "Checking analyst sentiment",
  search_global_macro_impact: "Scanning global macro impact",
};

// One thread_id per conversation — reused across turns so the agent
// remembers prior context (same pattern as the notebooks' short-term memory
// demo). "New chat" generates a fresh one and clears the visible history.
let threadId = crypto.randomUUID();

function attachChipHandlers() {
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.q;
      form.requestSubmit();
    });
  });
}
attachChipHandlers();

newChatBtn.addEventListener("click", () => {
  threadId = crypto.randomUUID();
  chatMessages.innerHTML = "";
  chatMessages.appendChild(emptyState);
  emptyState.classList.remove("hidden");
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;
  input.value = "";
  await sendMessage(query);
});

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendUserBubble(text) {
  emptyState.classList.add("hidden");
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `<div class="bubble user"></div>`;
  row.querySelector(".bubble").textContent = text;
  chatMessages.appendChild(row);
  scrollToBottom();
}

function appendAssistantBubble() {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.innerHTML = `
    <div class="bubble assistant">
      <ol class="thinking-steps"></ol>
      <div class="bubble-content"></div>
      <div class="bubble-meta hidden"></div>
    </div>
  `;
  chatMessages.appendChild(row);
  scrollToBottom();
  return {
    bubble: row.querySelector(".bubble"),
    steps: row.querySelector(".thinking-steps"),
    content: row.querySelector(".bubble-content"),
    meta: row.querySelector(".bubble-meta"),
  };
}

function addStep(stepsEl, text) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="dot"></span><span>${text}</span>`;
  stepsEl.appendChild(li);
  scrollToBottom();
}

function markLastDone(stepsEl) {
  const items = stepsEl.querySelectorAll("li");
  if (items.length) items[items.length - 1].classList.add("done");
}

async function sendMessage(query) {
  appendUserBubble(query);
  const ui = appendAssistantBubble();
  submitBtn.disabled = true;
  addStep(ui.steps, "Sending request...");

  try {
    const res = await fetch("/api/v1/analyze/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, thread_id: threadId }),
    });

    if (!res.ok || !res.body) {
      markLastDone(ui.steps);
      addStep(ui.steps, "Request failed. Please try again.");
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;
        try {
          handleEvent(ui, JSON.parse(jsonStr));
        } catch (err) {
          console.error("Failed to parse event", err, jsonStr);
        }
      }
    }
  } catch (err) {
    markLastDone(ui.steps);
    addStep(ui.steps, "Connection error. Is the server running?");
    console.error(err);
  } finally {
    submitBtn.disabled = false;
  }
}

function handleEvent(ui, event) {
  markLastDone(ui.steps);

  switch (event.type) {
    case "status":
      addStep(ui.steps, event.message);
      break;

    case "tool_start": {
      const label = TOOL_LABELS[event.tool] || `Calling ${event.tool}`;
      addStep(ui.steps, `${label} <code>${event.tool}</code>`);
      break;
    }

    case "tool_end":
      markLastDone(ui.steps);
      break;

    case "guardrail":
      break;

    case "blocked":
      ui.bubble.classList.add("blocked");
      ui.steps.remove();
      ui.content.innerHTML = `<strong>Guardrail triggered (${event.stage}):</strong> ${event.message}`;
      break;

    case "error":
      addStep(ui.steps, event.message);
      break;

    case "final":
      markLastDone(ui.steps);
      renderFinal(ui, event.data);
      break;
  }
  scrollToBottom();
}

function renderFinal(ui, data) {
  ui.steps.remove();

  if (data.status === "blocked") {
    ui.bubble.classList.add("blocked");
    ui.content.innerHTML = `<strong>Guardrail triggered (${data.guardrail_events[0]?.stage || ""}):</strong> ${data.answer}`;
    return;
  }

  const rawHtml = marked.parse(data.answer || "");
  ui.content.innerHTML = DOMPurify.sanitize(rawHtml);

  const toolsText = data.tools_used.length ? `${data.tools_used.length} tools used` : "No tools used";
  const traceText = `trace: ${data.trace_id.slice(0, 8)}`;

  ui.meta.classList.remove("hidden");
  ui.meta.innerHTML = `
    <span class="meta-badge">${toolsText}</span>
    <span class="meta-badge trace">${traceText}</span>
  `;
  (data.guardrail_events || []).forEach((ev) => {
    const chip = document.createElement("span");
    chip.className = "gr-chip" + (ev.result === "flagged" ? " flagged" : "");
    chip.textContent = `${ev.stage}: ${ev.result}`;
    ui.meta.appendChild(chip);
  });

  scrollToBottom();
}
