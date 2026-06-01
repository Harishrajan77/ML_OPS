import json
import os
import pickle
import socket
import sys
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


APP_HOST = "127.0.0.1"
APP_PORT = 8017

FEATURES = [
    ("Hour", "Hour", "0-23", 12),
    ("Occupied_Slots", "Occupied slots", "Currently filled spaces", 45),
    ("Available_Slots", "Available slots", "Currently open spaces", 55),
    ("Vehicle_Count", "Vehicle volume", "Detected vehicle flow", 100),
    ("Entry_Count", "Entries", "Incoming vehicles", 30),
    ("Exit_Count", "Exits", "Outgoing vehicles", 25),
]

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "day_04",
        "task_06",
        "parking_multiple_linear_regression_model.pkl",
    )
)


def install_sklearn_compat() -> None:
    if "sklearn.linear_model._base" in sys.modules:
        return

    sklearn_module = types.ModuleType("sklearn")
    linear_model_module = types.ModuleType("sklearn.linear_model")
    base_module = types.ModuleType("sklearn.linear_model._base")

    class LinearRegression:
        def predict(self, rows):
            predictions = []
            for row in rows:
                total = float(getattr(self, "intercept_", 0.0))
                coefficients = list(getattr(self, "coef_", []))
                total += sum(float(coef) * float(value) for coef, value in zip(coefficients, row))
                predictions.append(total)
            return predictions

    base_module.LinearRegression = LinearRegression
    linear_model_module.LinearRegression = LinearRegression
    linear_model_module._base = base_module
    sklearn_module.linear_model = linear_model_module

    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.linear_model"] = linear_model_module
    sys.modules["sklearn.linear_model._base"] = base_module


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    install_sklearn_compat()
    with open(MODEL_PATH, "rb") as handle:
        return pickle.load(handle)


MODEL = load_model()


def predict_occupancy(payload: dict) -> dict:
    values = []
    for key, label, _hint, _default in FEATURES:
        try:
            value = float(payload.get(key, ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a valid number.") from exc
        values.append(value)

    raw_prediction = float(MODEL.predict([values])[0])
    prediction = max(0.0, min(100.0, raw_prediction))
    status, tone, recommendation = classify_prediction(prediction)

    data = dict(zip([feature[0] for feature in FEATURES], values))
    occupied = max(0.0, data["Occupied_Slots"])
    available = max(0.0, data["Available_Slots"])
    capacity = occupied + available
    utilization_gap = max(0.0, 100.0 - prediction)

    return {
        "prediction": round(prediction, 2),
        "rawPrediction": round(raw_prediction, 4),
        "status": status,
        "tone": tone,
        "recommendation": recommendation,
        "capacity": round(capacity),
        "occupied": round(occupied),
        "available": round(available),
        "utilizationGap": round(utilization_gap, 2),
        "entryExitDelta": round(data["Entry_Count"] - data["Exit_Count"]),
        "clamped": raw_prediction != prediction,
    }


def classify_prediction(prediction: float) -> tuple[str, str, str]:
    if prediction < 35:
        return (
            "Open capacity",
            "success",
            "Keep standard routing active. Capacity is comfortable for incoming vehicles.",
        )
    if prediction < 70:
        return (
            "Balanced demand",
            "warning",
            "Monitor arrivals and departures. Demand is healthy without immediate overflow pressure.",
        )
    return (
        "High demand",
        "danger",
        "Prepare overflow guidance and surface nearby alternatives before queues form.",
    )


def app_html() -> str:
    defaults = {key: default for key, _label, _hint, default in FEATURES}
    fields = [
        {"key": key, "label": label, "hint": hint, "default": default}
        for key, label, hint, default in FEATURES
    ]
    return HTML.replace("__APP_DATA__", json.dumps({"defaults": defaults, "fields": fields}))


def find_available_port(start_port: int = APP_PORT, attempts: int = 20) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((APP_HOST, port))
            except OSError:
                continue
            return port
    raise OSError(f"No available local port found from {start_port} to {start_port + attempts - 1}.")


class ProductHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self.send_text(app_html(), "text/html; charset=utf-8")
            return
        if path == "/health":
            self.send_json({"ok": True})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/predict":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            result = predict_occupancy(payload)
            self.send_json({"ok": True, "result": result})
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            self.send_json({"ok": False, "error": f"Prediction failed: {exc}"}, status=500)

    def log_message(self, _format: str, *_args) -> None:
        return

    def send_text(self, content: str, content_type: str, status: int = 200) -> None:
        encoded = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_json(self, payload: dict, status: int = 200) -> None:
        self.send_text(json.dumps(payload), "application/json; charset=utf-8", status)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ParkSight Control</title>
  <style>
    :root {
      --bg: #eef3f6;
      --panel: #ffffff;
      --panel-2: #f7fafb;
      --ink: #10202f;
      --muted: #607183;
      --line: #d7e2e9;
      --teal: #0f766e;
      --teal-2: #0b5f59;
      --blue: #2563eb;
      --green: #16a34a;
      --amber: #d97706;
      --red: #dc2626;
      --shadow: 0 24px 70px rgba(25, 42, 62, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(135deg, rgba(15, 118, 110, 0.08), transparent 34%),
        linear-gradient(315deg, rgba(37, 99, 235, 0.08), transparent 36%),
        var(--bg);
      font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }

    button,
    input {
      font: inherit;
    }

    .shell {
      width: min(1440px, calc(100% - 40px));
      margin: 0 auto;
      padding: 28px 0;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 22px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 260px;
    }

    .mark {
      width: 44px;
      height: 44px;
      display: grid;
      place-items: center;
      color: white;
      background: #0f766e;
      border-radius: 8px;
      box-shadow: 0 12px 26px rgba(15, 118, 110, 0.25);
    }

    .mark svg {
      width: 25px;
      height: 25px;
    }

    h1,
    h2,
    p {
      margin: 0;
    }

    .brand h1 {
      font-size: 24px;
      line-height: 1;
      letter-spacing: 0;
    }

    .brand p {
      color: var(--muted);
      margin-top: 5px;
      font-size: 13px;
    }

    .nav {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(215, 226, 233, 0.9);
      border-radius: 8px;
      padding: 7px;
    }

    .nav span {
      color: var(--ink);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 13px;
      font-weight: 700;
    }

    .layout {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 22px;
      align-items: stretch;
    }

    .panel {
      background: rgba(255, 255, 255, 0.93);
      border: 1px solid rgba(215, 226, 233, 0.95);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .command {
      padding: 28px;
      display: grid;
      gap: 24px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 24px;
      align-items: start;
    }

    .eyebrow {
      color: var(--teal);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 9px;
    }

    .hero h2 {
      max-width: 720px;
      font-size: clamp(32px, 4vw, 58px);
      line-height: 1.02;
      letter-spacing: 0;
    }

    .hero p {
      max-width: 680px;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.65;
      margin-top: 14px;
    }

    .status-card {
      min-width: 220px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }

    .status-card .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .status-card .value {
      display: block;
      margin-top: 8px;
      font-size: 42px;
      line-height: 1;
      font-weight: 800;
      color: var(--teal);
    }

    .status-card .state {
      display: block;
      margin-top: 9px;
      color: var(--muted);
      font-weight: 700;
    }

    .map-wrap {
      display: grid;
      grid-template-columns: 1fr 260px;
      gap: 18px;
      align-items: stretch;
    }

    .parking-visual {
      min-height: 354px;
      padding: 18px;
      background:
        linear-gradient(90deg, rgba(16, 32, 47, 0.05) 1px, transparent 1px),
        linear-gradient(0deg, rgba(16, 32, 47, 0.05) 1px, transparent 1px),
        #f9fbfc;
      background-size: 28px 28px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      position: relative;
    }

    .parking-visual::before {
      content: "";
      position: absolute;
      inset: 50% 18px auto 18px;
      height: 46px;
      transform: translateY(-50%);
      border-top: 2px dashed #b8c8d3;
      border-bottom: 2px dashed #b8c8d3;
      background: rgba(238, 243, 246, 0.78);
    }

    .bay-grid {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: repeat(8, minmax(28px, 1fr));
      gap: 11px;
      height: 100%;
    }

    .slot {
      min-height: 58px;
      border: 1px solid #ccd9e1;
      border-radius: 7px;
      background: rgba(255, 255, 255, 0.82);
      position: relative;
      overflow: hidden;
    }

    .slot::after {
      content: "";
      position: absolute;
      left: 18%;
      right: 18%;
      top: 18%;
      bottom: 18%;
      border-radius: 7px;
      background: #dfe8ee;
      opacity: 0;
      transition: 180ms ease;
    }

    .slot.occupied::after {
      opacity: 1;
      background: var(--teal);
      box-shadow: inset 0 -10px 0 rgba(0, 0, 0, 0.09);
    }

    .slot.hot::after {
      background: var(--red);
    }

    .slot.medium::after {
      background: var(--amber);
    }

    .metrics {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }

    .metric {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .metric strong {
      display: block;
      font-size: 27px;
      margin-top: 8px;
    }

    .recommendation {
      display: grid;
      grid-template-columns: 44px 1fr;
      gap: 14px;
      align-items: center;
      background: #eff8f6;
      border: 1px solid #cfe8e4;
      border-radius: 8px;
      padding: 18px;
      color: #164e49;
      line-height: 1.55;
    }

    .recommendation svg {
      width: 27px;
      height: 27px;
    }

    .controls {
      padding: 24px;
      display: flex;
      flex-direction: column;
      min-height: 100%;
    }

    .controls h2 {
      font-size: 24px;
      margin-bottom: 6px;
    }

    .controls > p {
      color: var(--muted);
      line-height: 1.55;
      margin-bottom: 20px;
    }

    .form {
      display: grid;
      gap: 14px;
    }

    .field {
      display: grid;
      grid-template-columns: 1fr 136px;
      gap: 12px;
      align-items: center;
      padding: 12px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    .field label {
      display: block;
      font-weight: 800;
      margin-bottom: 4px;
    }

    .field small {
      color: var(--muted);
      line-height: 1.35;
    }

    .field input {
      width: 100%;
      border: 1px solid #cbd9e2;
      background: #fff;
      color: var(--ink);
      border-radius: 7px;
      padding: 12px 10px;
      outline: none;
      font-weight: 800;
      text-align: right;
    }

    .field input:focus {
      border-color: var(--teal);
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.13);
    }

    .actions {
      display: flex;
      gap: 10px;
      margin-top: 18px;
    }

    .btn {
      min-height: 46px;
      border: 0;
      border-radius: 8px;
      padding: 0 17px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 9px;
      font-weight: 800;
    }

    .btn-primary {
      background: var(--teal);
      color: #fff;
      flex: 1;
    }

    .btn-primary:hover {
      background: var(--teal-2);
    }

    .btn-secondary {
      background: var(--panel-2);
      color: var(--ink);
      border: 1px solid var(--line);
    }

    .btn svg {
      width: 18px;
      height: 18px;
    }

    .fineprint {
      margin-top: auto;
      padding-top: 18px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
    }

    .error {
      display: none;
      margin-top: 14px;
      padding: 12px;
      color: #7f1d1d;
      background: #fff1f2;
      border: 1px solid #fecdd3;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 700;
    }

    .danger {
      color: var(--red) !important;
    }

    .warning {
      color: var(--amber) !important;
    }

    .success {
      color: var(--green) !important;
    }

    @media (max-width: 1040px) {
      .layout,
      .map-wrap,
      .hero {
        grid-template-columns: 1fr;
      }

      .status-card {
        min-width: 0;
      }
    }

    @media (max-width: 720px) {
      .shell {
        width: min(100% - 24px, 1440px);
        padding: 16px 0;
      }

      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }

      .nav {
        width: 100%;
        overflow-x: auto;
      }

      .command,
      .controls {
        padding: 18px;
      }

      .field {
        grid-template-columns: 1fr;
      }

      .field input {
        text-align: left;
      }

      .actions {
        flex-direction: column;
      }

      .bay-grid {
        grid-template-columns: repeat(4, minmax(38px, 1fr));
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <section class="brand" aria-label="ParkSight Control">
        <div class="mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
            <path d="M6 20V4h6.3a5 5 0 0 1 0 10H9" />
            <path d="M9 20v-6" />
          </svg>
        </div>
        <div>
          <h1>ParkSight Control</h1>
          <p>Parking intelligence and occupancy forecasting</p>
        </div>
      </section>
      <nav class="nav" aria-label="Product navigation">
        <span>Forecast</span>
        <b>Demand</b>
        <b>Capacity</b>
        <b>Operations</b>
      </nav>
    </header>

    <section class="layout">
      <article class="panel command">
        <div class="hero">
          <div>
            <div class="eyebrow">Live occupancy engine</div>
            <h2>Predict parking demand before the lot feels crowded.</h2>
            <p>Use current traffic movement and slot availability to forecast occupancy, assess demand pressure, and guide smarter parking operations.</p>
          </div>
          <aside class="status-card" aria-live="polite">
            <span class="label">Forecast</span>
            <strong class="value" id="prediction">--%</strong>
            <span class="state" id="status">Waiting for input</span>
          </aside>
        </div>

        <div class="map-wrap">
          <section class="parking-visual" aria-label="Parking occupancy visual">
            <div class="bay-grid" id="bayGrid"></div>
          </section>
          <section class="metrics" aria-label="Parking metrics">
            <div class="metric">
              <span>Total capacity</span>
              <strong id="capacity">--</strong>
            </div>
            <div class="metric">
              <span>Available slots</span>
              <strong id="available">--</strong>
            </div>
            <div class="metric">
              <span>Entry exit delta</span>
              <strong id="delta">--</strong>
            </div>
          </section>
        </div>

        <section class="recommendation">
          <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 3v4" />
            <path d="M12 17v4" />
            <path d="M4.22 5.64l2.83 2.83" />
            <path d="M16.95 15.54l2.83 2.82" />
            <path d="M3 12h4" />
            <path d="M17 12h4" />
            <path d="M4.22 18.36l2.83-2.82" />
            <path d="M16.95 8.46l2.83-2.82" />
          </svg>
          <p id="recommendation">Enter values to generate an operational recommendation.</p>
        </section>
      </article>

      <aside class="panel controls">
        <h2>Forecast inputs</h2>
        <p>Update the operating snapshot and run the model. Values are sent locally to the Python prediction server.</p>
        <form class="form" id="predictForm"></form>
        <div class="actions">
          <button class="btn btn-primary" type="button" id="predictButton">
            <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
              <path d="M4 19V5" />
              <path d="M4 19h16" />
              <path d="M7 15l4-4 3 3 5-7" />
            </svg>
            Predict demand
          </button>
          <button class="btn btn-secondary" type="button" id="resetButton" title="Reset inputs">
            <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 12a9 9 0 1 0 3-6.7" />
              <path d="M3 4v5h5" />
            </svg>
          </button>
        </div>
        <div class="error" id="errorBox"></div>
        <p class="fineprint">The interface uses the saved regression model on your machine. No external service or internet connection is required for prediction.</p>
      </aside>
    </section>
  </main>

  <script>
    const appData = __APP_DATA__;
    const form = document.querySelector("#predictForm");
    const errorBox = document.querySelector("#errorBox");
    const prediction = document.querySelector("#prediction");
    const statusLabel = document.querySelector("#status");
    const recommendation = document.querySelector("#recommendation");
    const capacity = document.querySelector("#capacity");
    const available = document.querySelector("#available");
    const delta = document.querySelector("#delta");
    const bayGrid = document.querySelector("#bayGrid");

    function iconField(name) {
      const icons = {
        Hour: "clock",
        Occupied_Slots: "park",
        Available_Slots: "open",
        Vehicle_Count: "flow",
        Entry_Count: "in",
        Exit_Count: "out"
      };
      return icons[name] || "dot";
    }

    function createFields() {
      form.innerHTML = appData.fields.map((field) => `
        <div class="field">
          <div>
            <label for="${field.key}">${field.label}</label>
            <small>${field.hint}</small>
          </div>
          <input
            id="${field.key}"
            name="${field.key}"
            type="number"
            step="any"
            inputmode="decimal"
            value="${field.default}"
            data-icon="${iconField(field.key)}"
          />
        </div>
      `).join("");
      form.querySelectorAll("input").forEach((input) => {
        input.addEventListener("keydown", (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            runPrediction();
          }
        });
      });
    }

    function readPayload() {
      const payload = {};
      for (const field of appData.fields) {
        payload[field.key] = document.querySelector(`#${field.key}`).value;
      }
      return payload;
    }

    function setError(message) {
      errorBox.textContent = message || "";
      errorBox.style.display = message ? "block" : "none";
    }

    function toneClass(tone) {
      return tone === "danger" ? "danger" : tone === "warning" ? "warning" : "success";
    }

    function renderSlots(percent, tone) {
      const slotCount = 48;
      const filled = Math.round(slotCount * (percent / 100));
      const colorClass = tone === "danger" ? "hot" : tone === "warning" ? "medium" : "";
      bayGrid.innerHTML = "";
      for (let i = 0; i < slotCount; i += 1) {
        const slot = document.createElement("span");
        slot.className = `slot ${i < filled ? `occupied ${colorClass}` : ""}`;
        bayGrid.appendChild(slot);
      }
    }

    function renderResult(result) {
      const cls = toneClass(result.tone);
      prediction.className = `value ${cls}`;
      statusLabel.className = `state ${cls}`;
      prediction.textContent = `${result.prediction.toFixed(2)}%`;
      statusLabel.textContent = result.status;
      capacity.textContent = result.capacity;
      available.textContent = result.available;
      delta.textContent = result.entryExitDelta > 0 ? `+${result.entryExitDelta}` : result.entryExitDelta;
      recommendation.textContent = result.recommendation + (result.clamped ? " Display has been normalized to the 0-100% range." : "");
      renderSlots(result.prediction, result.tone);
    }

    async function runPrediction() {
      setError("");
      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(readPayload())
        });
        const data = await response.json();
        if (!data.ok) {
          throw new Error(data.error || "Prediction failed.");
        }
        renderResult(data.result);
      } catch (error) {
        setError(error.message);
      }
    }

    function resetFields() {
      for (const field of appData.fields) {
        document.querySelector(`#${field.key}`).value = appData.defaults[field.key];
      }
      runPrediction();
    }

    document.querySelector("#predictButton").addEventListener("click", runPrediction);
    document.querySelector("#resetButton").addEventListener("click", resetFields);

    createFields();
    runPrediction();
  </script>
</body>
</html>
"""


def main() -> None:
    port = find_available_port()
    server = ThreadingHTTPServer((APP_HOST, port), ProductHandler)
    print(f"ParkSight Control is running at http://{APP_HOST}:{port}")
    print("Press Ctrl+C to stop the server.")
    server.serve_forever()


if __name__ == "__main__":
    main()
