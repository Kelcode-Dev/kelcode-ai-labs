const form        = document.getElementById("rag-form");
const submitBtn   = document.querySelector("#rag-form button[type=submit]");
const answerBox   = document.getElementById("answer");
const sourcesList = document.getElementById("sources");
const debugCheck  = document.getElementById("debug-check");
const debugChunks = document.getElementById("debug-chunks");
const chunksDiv   = document.getElementById("chunks");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = document.getElementById("question").value.trim();
  if (!question) return;

  // ---- disable button + swap text ----
  submitBtn.disabled = true;
  const originalLabel = submitBtn.textContent;
  submitBtn.textContent = "Waiting…";

  answerBox.textContent      = "Thinking...";
  sourcesList.innerHTML      = "";
  chunksDiv.innerHTML        = "";
  debugChunks.style.display  = "none";

  try {
    // qa endpoint
    const qaRes = await fetch("/qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    const qaData = await qaRes.json();
    answerBox.innerHTML = marked.parse(qaData.answer);

    // dedupe sources
    const seen = new Set();
    const uniqueSources = qaData.sources.filter(src => {
      const key = `${src.title}|${src.source}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    uniqueSources.forEach(src => {
      const li = document.createElement("li");
      li.className   = "list-group-item";
      li.textContent = `${src.title} (${src.source})`;
      sourcesList.appendChild(li);
    });

    // chunks endpoint ===
    if (debugCheck.checked) {
      debugChunks.style.display = "block";
      const chunkRes = await fetch("/chunks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });
      const chunkData = await chunkRes.json();
      chunkData.chunks.forEach(chunk => {
        const pre = document.createElement("pre");
        pre.className  = "small p-2 border bg-light mb-2";
        pre.textContent = `#${chunk.rank} – ${chunk.title} (${chunk.source})\n\n${chunk.text}`;
        chunksDiv.appendChild(pre);
      });
    }
  } catch (err) {
    console.error(err);
    answerBox.textContent = "Something went wrong, check console.";
  } finally {
    // ---- re-enable button + restore text ----
    submitBtn.disabled = false;
    submitBtn.textContent = originalLabel;
  }
});
