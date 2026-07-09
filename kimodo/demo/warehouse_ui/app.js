const API_BASE = window.location.port === "8787" ? "" : "http://localhost:8787";

const INVENTORY = {
  coconut: { name: "Coconut", rack: "rack1", shelf: 1, stock: 15, icon: "CO", color: "#8b5a2b", pattern: "coconuts?" },
  tomato: { name: "Tomato", rack: "rack1", shelf: 2, stock: 20, icon: "TO", color: "#dc3c3c", pattern: "tomato(?:es)?" },
  curry_leaves: { name: "Curry leaves", rack: "rack1", shelf: 5, stock: 18, icon: "CL", color: "#3d9b50", pattern: "curry\\s+(?:leaves|leaf)" },
  potato: { name: "Potato", rack: "rack1", shelf: 4, stock: 20, icon: "PO", color: "#a77b45", pattern: "potato(?:es)?" },
  water: { name: "Water bottle", rack: "rack1", shelf: 3, stock: 12, icon: "W", color: "#177a94", pattern: "(?:water\\s+bottles?|bottles?\\s+of\\s+water|bottled\\s+water|water)" },
  rice: { name: "Rice packet", rack: "rack2", shelf: 1, stock: 16, icon: "RI", color: "#d8c39b", pattern: "(?:rice(?:\\s+(?:bags?|packets?))?|(?:bags?|packets?)\\s+of\\s+rice)" },
  red_gram: { name: "Red gram packet", rack: "rack2", shelf: 2, stock: 14, icon: "RG", color: "#c26a43", pattern: "(?:red\\s+grams?|redgram|toor\\s+dal|pigeon\\s+peas?)" },
  coke: { name: "Coke", rack: "rack2", shelf: 3, stock: 10, icon: "C", color: "#b92735", pattern: "(?:coke\\s+(?:bottles?|cans?)|(?:bottles?|cans?)\\s+of\\s+(?:coke|cola)|cokes?|colas?)" },
  ice_cream: { name: "Ice cream", rack: "rack2", shelf: 4, stock: 8, icon: "IC", color: "#a56ac4", pattern: "ice\\s+creams?" },
  chips: { name: "Chips packet", rack: "rack2", shelf: 5, stock: 18, icon: "CH", color: "#d99c27", pattern: "(?:chips?(?:\\s+packets?)?|packets?\\s+of\\s+chips?)" },
};

const BASE_MOTIONS = {
  "start:rack1": { stem: "go_to_rack1", seconds: 20.43 },
  "start:rack2": { stem: "go_to_rack2", seconds: 26.50 },
  "rack1:rack2": { stem: "go_rack1_to_rack2", seconds: 13.57 },
  "rack2:rack1": { stem: "go_rack2_to_rack1", seconds: 13.57 },
  "rack1:counter": { stem: "go_rack1_to_counter", seconds: 20.53 },
  "rack2:counter": { stem: "go_rack2_to_counter", seconds: 26.53 },
};

const T3_COMBINED_MOTIONS = {
  pick: "pick_item",
  place: "drop_item",
};

const state = {
  order: null,
  packingPlan: null,
  plan: [],
  running: false,
  stopRequested: false,
  previewedPlanSignature: null,
};

const el = {
  request: document.querySelector("#request-input"),
  planButton: document.querySelector("#plan-button"),
  runButton: document.querySelector("#run-button"),
  stopButton: document.querySelector("#stop-button"),
  hardwareButton: document.querySelector("#hardware-button"),
  hardwareConfirm: document.querySelector("#hardware-confirm"),
  emergencyStopButton: document.querySelector("#emergency-stop-button"),
  planList: document.querySelector("#plan-list"),
  planDuration: document.querySelector("#plan-duration"),
  orderMessage: document.querySelector("#order-message"),
  executionMessage: document.querySelector("#execution-message"),
  progress: document.querySelector("#progress-bar"),
  connectionDot: document.querySelector("#connection-dot"),
  connectionLabel: document.querySelector("#connection-label"),
  inventoryGrid: document.querySelector("#inventory-grid"),
};

function currentPlanSignature() {
  if (!state.order || !state.plan.length) return null;
  return JSON.stringify({
    order: state.order,
    motions: state.plan.map((step) => ({
      type: step.type,
      stem: step.stem || null,
      product: step.product || null,
      quantity: step.quantity || 0,
    })),
  });
}

function updateHardwareControls() {
  const previewIsCurrent = state.previewedPlanSignature === currentPlanSignature();
  el.hardwareConfirm.disabled = !previewIsCurrent || state.running;
  el.hardwareButton.disabled = !previewIsCurrent || !el.hardwareConfirm.checked || state.running;
}

function renderInventory() {
  el.inventoryGrid.innerHTML = Object.entries(INVENTORY).map(([productId, item]) => `
    <article class="inventory-card" style="--product-color: ${item.color}">
      <div class="product-icon">${item.icon}</div>
      <div>
        <strong>${item.name}</strong>
        <span>Rack ${item.rack === "rack1" ? "1" : "2"} · Shelf ${item.shelf}</span>
      </div>
      <div class="stock"><b data-stock="${productId}">${item.stock}</b><span>in stock</span></div>
    </article>
  `).join("");
}

function updateInventoryStocks() {
  el.inventoryGrid.querySelectorAll("[data-stock]").forEach((stockElement) => {
    stockElement.textContent = INVENTORY[stockElement.dataset.stock].stock;
  });
}

const NUMBER_VALUES = {
  a: 1, an: 1, zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5,
  six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11,
  twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16,
  seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20, thirty: 30,
  forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90,
};
const NUMBER_CORE = Object.keys(NUMBER_VALUES).join("|");
const QUANTITY_PATTERN = `(?:\\d+|(?:${NUMBER_CORE})(?:\\s+(?:and\\s+)?(?:${NUMBER_CORE}|hundred)){0,4})`;

function parseQuantity(value) {
  if (/^\d+$/.test(value)) return Number(value);
  let current = 0;
  value.split(/\s+/).filter((word) => word !== "and").forEach((word) => {
    if (word === "hundred") current = (current || 1) * 100;
    else current += NUMBER_VALUES[word] || 0;
  });
  return current;
}

function quantityForProduct(text, productPattern) {
  const before = text.match(new RegExp(`\\b(${QUANTITY_PATTERN})\\s+(?:${productPattern})\\b`, "i"));
  if (before) return parseQuantity(before[1].toLowerCase());

  const after = text.match(new RegExp(`\\b(?:${productPattern})\\s*(?:x|×)?\\s*(${QUANTITY_PATTERN})\\b`, "i"));
  if (after) return parseQuantity(after[1].toLowerCase());

  return new RegExp(`\\b(?:${productPattern})\\b`, "i").test(text) ? 1 : 0;
}

function parseOrder(text) {
  const normalized = text.trim().toLowerCase().replace(/-/g, " ").replace(/[^a-z0-9×\s]/g, " ");
  if (/\b(?:many|several|few|some)\b/.test(normalized)) {
    throw new Error("Please give an exact quantity, for example: 2 potatoes, 3 tomatoes and 1 ice cream.");
  }
  const order = {};
  Object.entries(INVENTORY).forEach(([productId, item]) => {
    const quantity = quantityForProduct(normalized, item.pattern);
    if (quantity > item.stock) throw new Error(`Only ${item.stock} × ${item.name} are in stock.`);
    if (quantity > 0) order[productId] = quantity;
  });
  if (!Object.keys(order).length) {
    throw new Error("No known item found. Choose any product shown in the ten-shelf inventory.");
  }
  return order;
}

function routeCost(route) {
  return route.slice(0, -1).reduce((sum, from, index) => {
    return sum + BASE_MOTIONS[`${from}:${route[index + 1]}`].seconds;
  }, 0);
}

async function requestPackingPlan(order) {
  const payload = await fetchJson("/packing-plan", {
    method: "POST",
    body: JSON.stringify({ order }),
  });
  return payload.plan;
}

function buildPlan(order, packedItems) {
  const steps = [];
  let location = "start";
  packedItems.forEach((packedItem, index) => {
    const item = INVENTORY[packedItem.product_id];
    if (!item) throw new Error(`OR-Tools returned unknown product: ${packedItem.product_id}`);
    if (location !== item.rack) {
      const motion = BASE_MOTIONS[`${location}:${item.rack}`];
      if (!motion) throw new Error(`No base motion is available from ${location} to ${item.rack}.`);
      steps.push({
        type: "navigate",
        title: `Navigate ${location} → ${item.rack}`,
        detail: `${motion.stem}_diff_drive.csv · ${motion.seconds.toFixed(1)}s`,
        ...motion,
      });
      location = item.rack;
    }
    const position = index + 1;
    steps.push({
      type: "pick",
      product: packedItem.product_id,
      quantity: 1,
      basketPosition: position,
      title: `Pick ${item.name} → basket position ${position}${position === 1 ? " (bottom)" : ""}`,
      detail: `Rack ${item.rack === "rack1" ? "1" : "2"} · Shelf ${item.shelf} · pick_item.csv`,
    });
  });
  const counterMotion = BASE_MOTIONS[`${location}:counter`];
  steps.push({
    type: "navigate",
    title: `Navigate ${location} → counter`,
    detail: `${counterMotion.stem}_diff_drive.csv · ${counterMotion.seconds.toFixed(1)}s`,
    ...counterMotion,
  });
  const quantity = Object.values(order).reduce((sum, value) => sum + value, 0);
  steps.push({
    type: "place",
    quantity,
    title: `Deliver basket with ${quantity} item${quantity > 1 ? "s" : ""} at counter`,
    detail: "drop_item.csv · synchronized T3 arms + base",
  });
  return steps;
}

function renderPlan() {
  el.planList.innerHTML = "";
  state.plan.forEach((step, index) => {
    const item = document.createElement("li");
    item.dataset.index = String(index);
    item.innerHTML = `<strong>${step.title}</strong><span>${step.detail}</span>`;
    el.planList.append(item);
  });
  const seconds = state.plan.filter((step) => step.type === "navigate").reduce((sum, step) => sum + step.seconds, 0);
  el.planDuration.textContent = `${seconds.toFixed(1)}s`;
  el.runButton.disabled = !state.plan.length || state.running;
  el.progress.style.width = "0%";
  updateHardwareControls();
}

async function createPlan() {
  state.previewedPlanSignature = null;
  el.hardwareConfirm.checked = false;
  el.planButton.disabled = true;
  el.orderMessage.className = "message";
  el.orderMessage.textContent = "OR-Tools is optimizing bottom-to-top packing and rack travel…";
  try {
    state.order = parseOrder(el.request.value);
    state.packingPlan = await requestPackingPlan(state.order);
    state.plan = buildPlan(state.order, state.packingPlan.items);
    renderPlan();
    const summary = Object.entries(state.order)
      .map(([productId, quantity]) => `${quantity} × ${INVENTORY[productId].name}`)
      .join(" + ");
    el.orderMessage.className = "message success";
    el.orderMessage.textContent = `Plan ready: ${summary}. OR-Tools packing order is bottom → top; safe packing takes priority over travel distance.`;
  } catch (error) {
    state.order = null;
    state.packingPlan = null;
    state.plan = [];
    el.orderMessage.className = "message error";
    el.orderMessage.textContent = error.message;
    el.planList.innerHTML = '<li class="empty-plan">No valid order to plan.</li>';
    el.planDuration.textContent = "—";
    el.runButton.disabled = true;
    updateHardwareControls();
  } finally {
    el.planButton.disabled = false;
  }
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

async function waitUntilStopped(statusLoader, timeoutMs = 120000) {
  const started = Date.now();
  await new Promise((resolve) => setTimeout(resolve, 250));
  while (!state.stopRequested) {
    const status = await statusLoader();
    if (!status.playing) return;
    if (Date.now() - started > timeoutMs) throw new Error("Preview timed out while waiting for the motion to finish.");
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("Preview stopped by user.");
}

async function playBase(stem, chain) {
  await fetchJson("/base-control", {
    method: "POST",
    body: JSON.stringify({ action: "preview", stem, chain }),
  });
  await waitUntilStopped(async () => {
    const payload = await fetchJson("/base-control");
    return payload.status;
  });
}

async function playCombinedT3(motion) {
  await fetchJson("/base-control", {
    method: "POST",
    body: JSON.stringify({ action: "t3-preview", motion }),
  });
  await waitUntilStopped(async () => {
    const payload = await fetchJson("/base-control");
    return payload.status;
  });
}

async function loadBaseForHardware(stem, chain) {
  await fetchJson("/base-control", {
    method: "POST",
    body: JSON.stringify({ action: "load", stem, chain }),
  });
}

async function loadCombinedT3ForHardware(motion) {
  await fetchJson("/base-control", {
    method: "POST",
    body: JSON.stringify({ action: "t3-load", motion }),
  });
}

async function playLoadedMotionOnHardware() {
  const started = await fetchJson("/robot-control", {
    method: "POST",
    body: JSON.stringify({
      action: "play",
      approved: true,
      dry_run: false,
      require_preview: true,
    }),
  });
  if (!started.status?.connected || !started.status?.playing) {
    throw new Error("T3 hardware stream did not enter the playing state.");
  }

  const startedAt = Date.now();
  while (!state.stopRequested) {
    const payload = await fetchJson("/robot-status");
    const status = payload.status;
    const output = Array.isArray(status.last_output) ? status.last_output.join("\n") : "";
    if (/\[ERROR\]|timed out|failed/i.test(output)) {
      throw new Error(output || "T3 hardware stream failed.");
    }
    if (!status.playing) return;
    if (Date.now() - startedAt > 180000) {
      throw new Error("T3 hardware playback timed out.");
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("Hardware execution stopped by user.");
}

function markStep(index, className) {
  const item = el.planList.querySelector(`[data-index="${index}"]`);
  if (item) item.classList.add(className);
}

async function runPreview() {
  if (!state.plan.length || state.running) return;
  state.previewedPlanSignature = null;
  el.hardwareConfirm.checked = false;
  state.running = true;
  state.stopRequested = false;
  el.runButton.disabled = true;
  el.stopButton.disabled = false;
  updateHardwareControls();
  el.executionMessage.className = "message";
  let firstNavigation = true;

  try {
    for (let index = 0; index < state.plan.length; index += 1) {
      if (state.stopRequested) throw new Error("Preview stopped by user.");
      const step = state.plan[index];
      markStep(index, "active");
      el.executionMessage.textContent = `Step ${index + 1}/${state.plan.length}: ${step.title}`;

      if (step.type === "navigate") {
        await playBase(step.stem, !firstNavigation);
        firstNavigation = false;
      } else if (step.type === "pick") {
        for (let count = 0; count < step.quantity; count += 1) {
          await playCombinedT3(T3_COMBINED_MOTIONS.pick);
        }
      } else if (step.type === "place") {
        for (let count = 0; count < step.quantity; count += 1) {
          await playCombinedT3(T3_COMBINED_MOTIONS.place);
        }
      }

      const item = el.planList.querySelector(`[data-index="${index}"]`);
      if (item) item.classList.remove("active");
      markStep(index, "done");
      el.progress.style.width = `${((index + 1) / state.plan.length) * 100}%`;
    }

    Object.entries(state.order).forEach(([productId, quantity]) => {
      INVENTORY[productId].stock -= quantity;
    });
    updateInventoryStocks();
    el.executionMessage.className = "message success";
    state.previewedPlanSignature = currentPlanSignature();
    el.hardwareConfirm.disabled = false;
    el.executionMessage.textContent = "Viser preview completed. Review the scene, then explicitly arm hardware execution.";
  } catch (error) {
    el.executionMessage.className = "message error";
    el.executionMessage.textContent = error.message;
  } finally {
    state.running = false;
    el.hardwareConfirm.checked = false;
    el.runButton.disabled = false;
    el.stopButton.disabled = true;
    updateHardwareControls();
  }
}

async function runHardware() {
  if (state.running) return;
  if (state.previewedPlanSignature !== currentPlanSignature()) {
    el.executionMessage.className = "message error";
    el.executionMessage.textContent = "Run the current plan completely in Viser before hardware execution.";
    return;
  }
  if (!el.hardwareConfirm.checked) return;

  el.hardwareConfirm.checked = false;
  state.running = true;
  state.stopRequested = false;
  el.runButton.disabled = true;
  el.stopButton.disabled = false;
  updateHardwareControls();
  el.executionMessage.className = "message";
  let firstNavigation = true;

  try {
    for (let index = 0; index < state.plan.length; index += 1) {
      if (state.stopRequested) throw new Error("Hardware execution stopped by user.");
      const step = state.plan[index];
      markStep(index, "active");
      el.executionMessage.textContent = `HARDWARE ${index + 1}/${state.plan.length}: ${step.title}`;

      if (step.type === "navigate") {
        await loadBaseForHardware(step.stem, !firstNavigation);
        firstNavigation = false;
        await playLoadedMotionOnHardware();
      } else if (step.type === "pick") {
        for (let count = 0; count < step.quantity; count += 1) {
          await loadCombinedT3ForHardware(T3_COMBINED_MOTIONS.pick);
          await playLoadedMotionOnHardware();
        }
      } else if (step.type === "place") {
        for (let count = 0; count < step.quantity; count += 1) {
          await loadCombinedT3ForHardware(T3_COMBINED_MOTIONS.place);
          await playLoadedMotionOnHardware();
        }
      }

      const item = el.planList.querySelector(`[data-index="${index}"]`);
      if (item) item.classList.remove("active");
      markStep(index, "done");
      el.progress.style.width = `${((index + 1) / state.plan.length) * 100}%`;
    }
    el.executionMessage.className = "message success";
    el.executionMessage.textContent = "T3 hardware plan completed. Arms and TaraBase were stopped safely.";
    el.hardwareConfirm.checked = false;
  } catch (error) {
    await Promise.allSettled([
      fetchJson("/base-control", { method: "POST", body: JSON.stringify({ action: "stop" }) }),
      fetchJson("/robot-control", { method: "POST", body: JSON.stringify({ action: "stop" }) }),
    ]);
    el.executionMessage.className = "message error";
    el.executionMessage.textContent = `Hardware stopped: ${error.message}`;
  } finally {
    state.running = false;
    el.runButton.disabled = false;
    el.stopButton.disabled = true;
    updateHardwareControls();
  }
}

async function stopPreview() {
  state.stopRequested = true;
  el.stopButton.disabled = true;
  await Promise.allSettled([
    fetchJson("/base-control", { method: "POST", body: JSON.stringify({ action: "stop" }) }),
    fetchJson("/robot-control", { method: "POST", body: JSON.stringify({ action: "stop" }) }),
  ]);
}

async function emergencyStop() {
  state.stopRequested = true;
  el.stopButton.disabled = true;
  el.executionMessage.className = "message error";
  el.executionMessage.textContent = "EMERGENCY STOP requested — stopping base and arm hardware…";
  const results = await Promise.allSettled([
    fetchJson("/base-control", { method: "POST", body: JSON.stringify({ action: "stop" }) }),
    fetchJson("/robot-control", { method: "POST", body: JSON.stringify({ action: "emergency-stop" }) }),
  ]);
  const failed = results.find((result) => result.status === "rejected");
  if (failed) {
    el.executionMessage.textContent = `EMERGENCY STOP sent, but confirmation failed: ${failed.reason?.message || failed.reason}`;
    return;
  }
  el.executionMessage.textContent = "EMERGENCY STOP confirmed: zero base RPM sent; arm stream disconnected.";
}

async function checkConnection() {
  try {
    const payload = await fetchJson("/status");
    const connected = payload.status.clients.length > 0;
    el.connectionDot.className = `status-dot ${connected ? "online" : "offline"}`;
    el.connectionLabel.textContent = connected ? "Viser connected · control ready" : "Open the Viser page to connect";
  } catch (_) {
    el.connectionDot.className = "status-dot offline";
    el.connectionLabel.textContent = "Kimodo control API offline";
  }
}

el.planButton.addEventListener("click", createPlan);
el.request.addEventListener("keydown", (event) => { if (event.key === "Enter") createPlan(); });
el.runButton.addEventListener("click", runPreview);
el.stopButton.addEventListener("click", stopPreview);
el.hardwareButton.addEventListener("click", runHardware);
el.hardwareConfirm.addEventListener("change", updateHardwareControls);
el.emergencyStopButton.addEventListener("click", emergencyStop);
document.querySelectorAll("[data-request]").forEach((button) => {
  button.addEventListener("click", () => {
    el.request.value = button.dataset.request;
    createPlan();
  });
});

renderInventory();
createPlan();
checkConnection();
setInterval(checkConnection, 5000);
