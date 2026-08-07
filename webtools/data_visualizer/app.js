const state = {
  datasets: [],
  selected: new Set(),
  summary: null,
};

const colors = {
  green: "#28785f",
  teal: "#0c7280",
  brick: "#a54635",
  amber: "#c9891d",
  violet: "#6556a8",
  ink: "#202124",
  muted: "#69706a",
  line: "#ded8cd",
  surface: "#fffdf8",
};

const accents = [colors.green, colors.teal, colors.brick, colors.amber, colors.violet];

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refreshButton").addEventListener("click", loadDatasets);
  document.getElementById("selectAllButton").addEventListener("click", () => {
    state.selected = new Set(state.datasets.map((item) => item.name));
    renderDatasetList();
    loadSummary();
  });
  document.getElementById("clearButton").addEventListener("click", () => {
    state.selected.clear();
    renderDatasetList();
    renderEmptyDashboard("请选择至少一个数据文件");
  });
  loadDatasets();
});

async function loadDatasets() {
  setStatus("扫描 data/");
  try {
    const response = await fetch("/api/datasets");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法读取数据文件");
    state.datasets = payload.datasets || [];
    state.selected = new Set(state.datasets.map((item) => item.name));
    renderDatasetList();
    if (state.datasets.length === 0) {
      renderEmptyDashboard("data/ 下没有 .pt 数据文件");
      return;
    }
    await loadSummary();
  } catch (error) {
    renderError(error);
  }
}

async function loadSummary() {
  const files = Array.from(state.selected);
  if (files.length === 0) {
    renderEmptyDashboard("请选择至少一个数据文件");
    return;
  }

  setStatus(`加载 ${files.length} 个文件`);
  const params = new URLSearchParams();
  for (const file of files) params.append("file", file);

  try {
    const response = await fetch(`/api/summary?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "统计失败");
    state.summary = payload;
    document.getElementById("dataDir").textContent = payload.dataDir || "data/";
    renderDashboard(payload);
    setStatus("已更新");
  } catch (error) {
    renderError(error);
  }
}

function renderDatasetList() {
  const list = document.getElementById("datasetList");
  if (state.datasets.length === 0) {
    list.innerHTML = `<div class="empty-state">没有找到 .pt 文件</div>`;
    return;
  }

  list.innerHTML = state.datasets
    .map((item) => {
      const checked = state.selected.has(item.name) ? "checked" : "";
      return `
        <label class="dataset-item">
          <input type="checkbox" data-file="${escapeHtml(item.name)}" ${checked} />
          <span>
            <span class="dataset-name">${escapeHtml(item.name)}</span>
            <span class="dataset-meta">${formatBytes(item.sizeBytes)} · ${formatDate(item.modified)}</span>
          </span>
        </label>
      `;
    })
    .join("");

  list.querySelectorAll("input[type='checkbox']").forEach((input) => {
    input.addEventListener("change", (event) => {
      const file = event.target.dataset.file;
      if (event.target.checked) state.selected.add(file);
      else state.selected.delete(file);
      loadSummary();
    });
  });
}

function renderDashboard(payload) {
  const combined = payload.combined;
  renderMetrics(combined);
  renderLimitations(combined.limitations || []);

  drawOutcomeChart("outcomeChart", combined.metrics.outcomes, [
    ["胜", colors.green],
    ["负", colors.brick],
    ["和", colors.amber],
  ]);
  drawOutcomeChart("firstMoverChart", combined.metrics.estimatedFirstMoverOutcomes, [
    ["先手胜", colors.teal],
    ["先手负", colors.brick],
    ["和", colors.amber],
  ]);
  drawBarChart("legalChart", combined.distributions.legalMoves, colors.green);
  drawBarChart("moveChart", combined.distributions.estimatedMoveIndex, colors.teal);
  drawBarChart("marginChart", combined.distributions.currentPieceDiff, colors.brick);
  drawBarChart("policyChart", combined.distributions.topPolicy, colors.violet);

  const heatmaps = combined.heatmaps;
  if (heatmaps) {
    drawHeatmap("policyHeatmap", heatmaps.policy, colors.violet);
    drawHeatmap("legalHeatmap", heatmaps.legal, colors.teal);
    drawHeatmap("ownHeatmap", heatmaps.own, colors.green);
    drawHeatmap("opponentHeatmap", heatmaps.opponent, colors.brick);
  } else {
    clearCanvas("policyHeatmap", "不同棋盘尺寸无法合并");
    clearCanvas("legalHeatmap", "不同棋盘尺寸无法合并");
    clearCanvas("ownHeatmap", "不同棋盘尺寸无法合并");
    clearCanvas("opponentHeatmap", "不同棋盘尺寸无法合并");
  }

  renderPhaseTable(combined.phaseSummary);
  renderDatasetTable(payload.datasets || []);
}

function renderMetrics(combined) {
  const metrics = [
    {
      label: "训练样本",
      value: formatNumber(combined.sampleCount),
      note: `${formatNumber(combined.datasetCount)} 个数据文件`,
    },
    {
      label: "当前玩家胜率",
      value: formatPercent(combined.metrics.outcomes.winRate),
      note: `${formatNumber(combined.metrics.outcomes.wins)} 胜 / ${formatNumber(combined.metrics.outcomes.losses)} 负`,
    },
    {
      label: "估算先手胜率",
      value: formatPercent(combined.metrics.estimatedFirstMoverOutcomes.winRate),
      note: "按样本所在估算手数奇偶换算",
    },
    {
      label: "平均合法步",
      value: formatFixed(combined.metrics.legalMoves.mean, 2),
      note: `中位数 ${formatFixed(combined.metrics.legalMoves.median, 1)}`,
    },
    {
      label: "平均当前目差",
      value: formatSigned(combined.metrics.currentPieceDiff.mean, 2),
      note: "当前视角，非终局目差",
    },
    {
      label: "平均估算手数",
      value: formatFixed(combined.metrics.estimatedMoveIndex.mean, 1),
      note: `最大 ${formatFixed(combined.metrics.estimatedMoveIndex.max, 0)}`,
    },
    {
      label: "策略最高概率",
      value: formatPercent(combined.metrics.topPolicy.mean),
      note: "MCTS 访问分布平均峰值",
    },
    {
      label: "无效/跳过步数",
      value: "不可用",
      note: "pass 未写入当前训练样本格式",
    },
  ];

  document.getElementById("metrics").innerHTML = metrics
    .map(
      (item, index) => `
        <article class="metric-card" style="--accent:${accents[index % accents.length]}">
          <div class="metric-label">${escapeHtml(item.label)}</div>
          <div class="metric-value">${escapeHtml(item.value)}</div>
          <div class="metric-note">${escapeHtml(item.note)}</div>
        </article>
      `,
    )
    .join("");
}

function renderLimitations(limitations) {
  const container = document.getElementById("limitations");
  if (!limitations.length) {
    container.innerHTML = `<div class="empty-state">当前数据字段完整</div>`;
    return;
  }
  container.innerHTML = limitations
    .map(
      (item) => `
        <div class="limitation">
          <strong>${escapeHtml(item.label)}</strong>
          <span>${escapeHtml(item.reason)}</span>
        </div>
      `,
    )
    .join("");
}

function renderPhaseTable(phaseSummary) {
  const rows = Object.values(phaseSummary || {});
  document.getElementById("phaseTable").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>阶段</th>
          <th>样本</th>
          <th>胜率</th>
          <th>负率</th>
          <th>和率</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
              <tr>
                <td>${escapeHtml(row.label)}</td>
                <td>${formatNumber(row.samples)}</td>
                <td>${formatPercent(row.outcomes.winRate)}</td>
                <td>${formatPercent(row.outcomes.lossRate)}</td>
                <td>${formatPercent(row.outcomes.drawRate)}</td>
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderDatasetTable(datasets) {
  document.getElementById("datasetTable").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>文件</th>
          <th>样本</th>
          <th>棋盘</th>
          <th>胜率</th>
          <th>平均合法步</th>
          <th>平均目差</th>
        </tr>
      </thead>
      <tbody>
        ${datasets
          .map(
            (item) => `
              <tr>
                <td>${escapeHtml(item.name)}</td>
                <td>${formatNumber(item.sampleCount)}</td>
                <td>${item.boardSize}x${item.boardSize}</td>
                <td>${formatPercent(item.metrics.outcomes.winRate)}</td>
                <td>${formatFixed(item.metrics.legalMoves.mean, 2)}</td>
                <td>${formatSigned(item.metrics.currentPieceDiff.mean, 2)}</td>
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function drawOutcomeChart(canvasId, outcomes, labels) {
  const data = [
    { label: labels[0][0], value: outcomes.wins || 0, color: labels[0][1] },
    { label: labels[1][0], value: outcomes.losses || 0, color: labels[1][1] },
    { label: labels[2][0], value: outcomes.draws || 0, color: labels[2][1] },
  ];
  drawHorizontalBars(canvasId, data);
}

function drawBarChart(canvasId, distribution, color) {
  const canvas = document.getElementById(canvasId);
  const ctx = setupCanvas(canvas);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight || Number(canvas.getAttribute("height"));
  ctx.clearRect(0, 0, width, height);

  if (!distribution || distribution.length === 0) {
    drawMessage(ctx, width, height, "暂无数据");
    return;
  }

  const padding = { top: 12, right: 10, bottom: 34, left: 40 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const maxValue = Math.max(...distribution.map((item) => item.value), 1);
  const step = Math.max(1, Math.ceil(distribution.length / 16));
  const barWidth = Math.max(2, chartWidth / distribution.length - 1);

  ctx.strokeStyle = colors.line;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + chartHeight);
  ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
  ctx.stroke();

  distribution.forEach((item, index) => {
    const x = padding.left + index * (chartWidth / distribution.length);
    const barHeight = (item.value / maxValue) * chartHeight;
    const y = padding.top + chartHeight - barHeight;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.88;
    ctx.fillRect(x, y, barWidth, barHeight);
    ctx.globalAlpha = 1;

    if (index % step === 0) {
      ctx.save();
      ctx.translate(x + barWidth / 2, padding.top + chartHeight + 18);
      ctx.rotate(-Math.PI / 5);
      ctx.fillStyle = colors.muted;
      ctx.font = "11px Segoe UI, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(item.label, 0, 0);
      ctx.restore();
    }
  });

  ctx.fillStyle = colors.muted;
  ctx.font = "12px Segoe UI, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText(formatNumber(maxValue), padding.left - 6, padding.top + 8);
  ctx.fillText("0", padding.left - 6, padding.top + chartHeight + 4);
}

function drawHorizontalBars(canvasId, data) {
  const canvas = document.getElementById(canvasId);
  const ctx = setupCanvas(canvas);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight || Number(canvas.getAttribute("height"));
  ctx.clearRect(0, 0, width, height);

  const total = data.reduce((sum, item) => sum + item.value, 0);
  if (!total) {
    drawMessage(ctx, width, height, "暂无数据");
    return;
  }

  const padding = 18;
  const barHeight = 34;
  const gap = 18;
  const chartWidth = width - padding * 2;
  let y = padding + 6;

  data.forEach((item) => {
    const rate = item.value / total;
    ctx.fillStyle = "#f0eadf";
    ctx.fillRect(padding, y, chartWidth, barHeight);
    ctx.fillStyle = item.color;
    ctx.fillRect(padding, y, chartWidth * rate, barHeight);
    ctx.fillStyle = colors.ink;
    ctx.font = "700 13px Segoe UI, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(item.label, padding, y - 5);
    ctx.textAlign = "right";
    ctx.fillText(`${formatPercent(rate)} · ${formatNumber(item.value)}`, padding + chartWidth, y - 5);
    y += barHeight + gap;
  });
}

function drawHeatmap(canvasId, matrix, accent) {
  const canvas = document.getElementById(canvasId);
  const ctx = setupCanvas(canvas, true);
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  if (!matrix || !matrix.length) {
    drawMessage(ctx, width, height, "暂无数据");
    return;
  }

  const rows = matrix.length;
  const cols = matrix[0].length;
  const maxValue = Math.max(...matrix.flat(), 1e-12);
  const cellW = width / cols;
  const cellH = height / rows;
  const rgb = hexToRgb(accent);

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const intensity = Math.sqrt(Math.max(0, matrix[row][col]) / maxValue);
      ctx.fillStyle = mixRgb([248, 244, 235], rgb, intensity);
      ctx.fillRect(col * cellW, row * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }

  ctx.strokeStyle = "rgba(32, 33, 36, 0.28)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= rows; i += 1) {
    const y = i * cellH;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  for (let i = 0; i <= cols; i += 1) {
    const x = i * cellW;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
}

function setupCanvas(canvas, square = false) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const cssWidth = Math.max(1, rect.width);
  const cssHeight = square ? cssWidth : Math.max(1, rect.height || Number(canvas.getAttribute("height")));
  canvas.width = Math.round(cssWidth * ratio);
  canvas.height = Math.round(cssHeight * ratio);
  canvas.style.height = `${cssHeight}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return ctx;
}

function clearCanvas(canvasId, message) {
  const canvas = document.getElementById(canvasId);
  const ctx = setupCanvas(canvas, true);
  drawMessage(ctx, canvas.clientWidth, canvas.clientHeight, message);
}

function drawMessage(ctx, width, height, message) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = colors.muted;
  ctx.font = "13px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, width / 2, height / 2);
}

function renderEmptyDashboard(message) {
  document.getElementById("metrics").innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
  ["outcomeChart", "firstMoverChart", "legalChart", "moveChart", "marginChart", "policyChart"].forEach((id) =>
    clearCanvas(id, message),
  );
  ["policyHeatmap", "legalHeatmap", "ownHeatmap", "opponentHeatmap"].forEach((id) => clearCanvas(id, message));
  document.getElementById("phaseTable").innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
  document.getElementById("datasetTable").innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
  setStatus(message);
}

function renderError(error) {
  const message = error instanceof Error ? error.message : String(error);
  renderEmptyDashboard(message);
  document.getElementById("limitations").innerHTML = `<div class="limitation"><strong>加载失败</strong><span>${escapeHtml(
    message,
  )}</span></div>`;
}

function setStatus(message) {
  document.getElementById("loadStatus").textContent = message;
}

function formatNumber(value) {
  return new Intl.NumberFormat("zh-CN").format(Number(value || 0));
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function formatFixed(value, digits) {
  return Number(value || 0).toFixed(digits);
}

function formatSigned(value, digits) {
  const number = Number(value || 0);
  return `${number > 0 ? "+" : ""}${number.toFixed(digits)}`;
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(seconds) {
  return new Date(Number(seconds || 0) * 1000).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  return [
    Number.parseInt(clean.slice(0, 2), 16),
    Number.parseInt(clean.slice(2, 4), 16),
    Number.parseInt(clean.slice(4, 6), 16),
  ];
}

function mixRgb(from, to, amount) {
  const clamped = Math.max(0, Math.min(1, amount));
  const mixed = from.map((value, index) => Math.round(value + (to[index] - value) * clamped));
  return `rgb(${mixed[0]}, ${mixed[1]}, ${mixed[2]})`;
}

window.addEventListener("resize", () => {
  if (state.summary) renderDashboard(state.summary);
});
