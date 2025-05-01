const form      = document.getElementById("classify-form");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  resultDiv.className = "result-card text-center";
  resultDiv.innerHTML = "⏳ Classifying…";

  const text  = document.getElementById("headline").value;
  const model = document.getElementById("model-select").value;
  const top5  = document.getElementById("top5-check").checked;
  const url   = top5 ? "/top5" : "/predict";

  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Model-Name": model,
      },
      body: JSON.stringify({ text }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    if (top5) {
      // render list of top5
        // build HTML for top5 micro-bars
        const html = data.top5.map(item => {
          const hue = Math.round(item.score * 120);
          const color = `hsl(${hue},100%,45%)`;
          const pct   = (item.score * 100).toFixed(1);

          return `
            <div class="top5-item">
              <div class="top5-label">${item.label} (${pct}%)</div>
              <div class="progress" style="height:0.75rem; border-radius:0.375rem;">
                <div
                  class="progress-bar"
                  role="progressbar"
                  style="
                    width: ${pct}%;
                    background-color: ${color};
                    transition: width 0.3s, background-color 0.3s;
                  "
                  aria-valuenow="${pct}"
                  aria-valuemin="0"
                  aria-valuemax="100"
                ></div>
              </div>
            </div>
          `;
        }).join("");

      resultDiv.classList.add("success");
      resultDiv.innerHTML = `<div class="top5-list">${html}</div>`;
    } else {
      // render single prediction + progress bar
      const { label, score } = data.category;
      resultDiv.classList.add("success");
        // compute a hue: 0° = red, 60° = yellow, 120° = green
        const hue = Math.round(score * 120);
        const heatColor = `hsl(${hue},100%,45%)`;

        resultDiv.innerHTML = `
          <h4 class="fw-bold mb-1">${label}</h4>
          <div class="progress" style="height: 1.5rem;">
            <div
              class="progress-bar heat-bar"
              role="progressbar"
              style="
                width: ${(score*100).toFixed(1)}%;
                background-color: ${heatColor};
              "
              aria-valuenow="${(score*100).toFixed(1)}"
              aria-valuemin="0"
              aria-valuemax="100"
            >
              ${(score*100).toFixed(1)}%
            </div>
          </div>
      `;
    }
  } catch (err) {
    resultDiv.classList.add("error");
    resultDiv.textContent = `⚠️ ${err.message || err}`;
  }
});
