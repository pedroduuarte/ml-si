const form = document.querySelector("#predictionForm");
const modelStatus = document.querySelector("#modelStatus");
const messageBox = document.querySelector("#messageBox");
const predictionLabel = document.querySelector("#predictionLabel");
const probabilityValue = document.querySelector("#probabilityValue");
const probabilityBar = document.querySelector("#probabilityBar");
const modelName = document.querySelector("#modelName");
const f1Score = document.querySelector("#f1Score");
const recallScore = document.querySelector("#recallScore");

const percentFormatter = new Intl.NumberFormat("pt-BR", {
  style: "percent",
  maximumFractionDigits: 1,
});

let debounceTimer = null;

function setMessage(text, isError = false) {
  messageBox.textContent = text;
  messageBox.classList.toggle("error", isError);
}

function collectPayload() {
  return Object.fromEntries(new FormData(form).entries());
}

function updateMetric(target, value) {
  target.textContent = typeof value === "number" ? percentFormatter.format(value) : "--";
}

function updateResult(data) {
  const probability = data.probability_churn;
  predictionLabel.textContent = data.label;
  probabilityValue.textContent =
    typeof probability === "number" ? percentFormatter.format(probability) : "--";
  probabilityBar.style.width =
    typeof probability === "number" ? `${Math.round(probability * 100)}%` : "0%";
  modelName.textContent = data.model_name || "Decision Tree";
  updateMetric(f1Score, data.metrics?.f1_score);
  updateMetric(recallScore, data.metrics?.recall);
}

async function refreshHealth() {
  const response = await fetch("/api/health");
  const health = await response.json();
  modelStatus.textContent = health.model_available ? "Modelo carregado" : "Modelo pendente";
  modelStatus.classList.toggle("ready", health.model_available);
  modelStatus.classList.toggle("pending", !health.model_available);
}

async function predict() {
  if (!form.reportValidity()) {
    return;
  }

  setMessage("Calculando predicao...");
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(collectPayload()),
  });
  const data = await response.json();

  if (!response.ok) {
    setMessage(data.setup || data.error || "Nao foi possivel prever.", true);
    predictionLabel.textContent = "Sem resultado";
    probabilityValue.textContent = "--";
    probabilityBar.style.width = "0%";
    return;
  }

  updateResult(data);
  setMessage("Predicao atualizada.");
}

function schedulePrediction() {
  window.clearTimeout(debounceTimer);
  debounceTimer = window.setTimeout(predict, 260);
}

form.addEventListener("input", schedulePrediction);
form.addEventListener("submit", (event) => {
  event.preventDefault();
  predict();
});

refreshHealth()
  .then(predict)
  .catch(() => {
    setMessage("Servidor local indisponivel.", true);
  });
