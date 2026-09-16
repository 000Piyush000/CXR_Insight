(() => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const imageFrame = document.getElementById("image-frame");
  const previewImg = document.getElementById("preview-img");
  const viewBadge = document.getElementById("view-badge");
  const findingsOverlay = document.getElementById("findings-overlay");

  const patientList = document.getElementById("patient-list");
  const patientPlaceholder = document.getElementById("patient-placeholder");

  const generateBtn = document.getElementById("generate-btn");
  const resetBtn = document.getElementById("reset-btn");
  const uploadError = document.getElementById("upload-error");

  const reportPlaceholder = document.getElementById("report-placeholder");
  const reportSections = document.getElementById("report-sections");
  const reportModeNote = document.getElementById("report-mode-note");
  const findingsText = document.getElementById("findings-text");
  const impressionHeading = document.getElementById("impression-heading");
  const impressionText = document.getElementById("impression-text");
  const rawReport = document.getElementById("raw-report");
  const reportBox = document.getElementById("report-box");

  const foundTagsBox = document.getElementById("found-tags");
  const tagsRowFound = document.getElementById("tags-row-found");
  const ruledoutTagsBox = document.getElementById("ruledout-tags");
  const tagsRowRuledout = document.getElementById("tags-row-ruledout");

  const searchInput = document.getElementById("disease-search-input");
  const searchBtn = document.getElementById("disease-search-btn");
  const downloadBtn = document.getElementById("download-btn");

  const chatWindow = document.getElementById("chat-window");
  const chatInput = document.getElementById("chat-input");
  const chatSend = document.getElementById("chat-send");
  const exampleQuestions = document.getElementById("example-questions");

  let sessionId = null;
  let lastMetadata = null;
  let lastReportText = null;

  // ---- Finding keyword patterns (used to build clickable web-search tags) ----
  const FINDING_PATTERNS = [
    { re: /pleural effusion/i, label: "Pleural Effusion" },
    { re: /pneumothorax/i, label: "Pneumothorax" },
    { re: /consolidation/i, label: "Consolidation" },
    { re: /cardiomegaly/i, label: "Cardiomegaly" },
    { re: /(bibasilar |basilar )?opacit(y|ies)/i, label: "Opacities" },
    { re: /(pulmonary )?edema/i, label: "Pulmonary Edema" },
    { re: /atelectasis/i, label: "Atelectasis" },
    { re: /(catheter|central venous line|picc line)/i, label: "Catheter / Line" },
    { re: /nodule/i, label: "Nodule" },
    { re: /infiltrate/i, label: "Infiltrate" },
    { re: /fracture/i, label: "Fracture" },
    { re: /emphysema/i, label: "Emphysema" },
    { re: /fibrosis/i, label: "Fibrosis" },
    { re: /\bmass\b/i, label: "Mass" },
  ];
  const NEGATION_RE = /\b(no|without|absence of|not|negative for|clear of|free of|no evidence of)\b/i;

  function analyzeFindings(text) {
    const sentences = text.split(/(?<=[.!?])\s+/);
    const found = [];
    const ruledOut = [];
    const seen = new Set();
    for (const sentence of sentences) {
      for (const { re, label } of FINDING_PATTERNS) {
        if (seen.has(label)) continue;
        const m = sentence.match(re);
        if (!m) continue;
        const before = sentence.slice(0, m.index).toLowerCase();
        seen.add(label);
        (NEGATION_RE.test(before) ? ruledOut : found).push(label);
      }
    }
    return { found, ruledOut };
  }

  function searchWeb(term) {
    const q = encodeURIComponent(`${term} radiology chest x-ray`);
    window.open(`https://www.google.com/search?q=${q}`, "_blank", "noopener");
  }

  function makeTagChip(label, ruledOut) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tag-chip" + (ruledOut ? " ruled-out" : "");
    btn.textContent = label;
    btn.title = `Search the web for "${label}"`;
    btn.addEventListener("click", () => searchWeb(label));
    return btn;
  }

  function renderFindingTags(text) {
    const { found, ruledOut } = analyzeFindings(text);

    tagsRowFound.innerHTML = "";
    findingsOverlay.innerHTML = "";
    if (found.length) {
      found.forEach((label) => {
        tagsRowFound.appendChild(makeTagChip(label, false));
        findingsOverlay.appendChild(makeTagChip(label, false));
      });
      foundTagsBox.hidden = false;
      findingsOverlay.hidden = false;
    } else {
      foundTagsBox.hidden = true;
      findingsOverlay.hidden = true;
    }

    tagsRowRuledout.innerHTML = "";
    if (ruledOut.length) {
      ruledOut.forEach((label) => tagsRowRuledout.appendChild(makeTagChip(label, true)));
      ruledoutTagsBox.hidden = false;
    } else {
      ruledoutTagsBox.hidden = true;
    }
  }

  function parseReport(text) {
    const findingsMatch = text.match(/FINDINGS[^:]*:\s*([\s\S]*?)(?:\n\s*\n\s*(?:IMPRESSION|NOTE)|$)/i);
    const impressionMatch = text.match(/IMPRESSION:\s*([\s\S]*?)(?:\n\s*\n\s*NOTE|$)/i);
    return {
      findings: findingsMatch ? findingsMatch[1].trim() : text.trim(),
      impression: impressionMatch ? impressionMatch[1].trim() : null,
    };
  }

  function setError(msg) {
    uploadError.textContent = msg || "";
  }

  function fillMetadata(meta) {
    document.getElementById("meta-patient").textContent = meta.patient_id;
    document.getElementById("meta-view").textContent = meta.view;
    document.getElementById("meta-age").textContent = meta.age;
    document.getElementById("meta-gender").textContent = meta.gender;
    document.getElementById("meta-ethnicity").textContent = meta.ethnicity;
    patientList.hidden = false;
    patientPlaceholder.hidden = true;
  }

  async function uploadFile(file) {
    setError("");
    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");

      sessionId = data.session_id;
      lastMetadata = data.metadata;
      previewImg.src = data.image_url;
      viewBadge.textContent = (data.metadata.view || "").toUpperCase();
      dropzone.hidden = true;
      imageFrame.hidden = false;
      fillMetadata(data.metadata);
      generateBtn.disabled = false;
    } catch (err) {
      setError(err.message);
    }
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) uploadFile(e.target.files[0]);
  });

  generateBtn.addEventListener("click", async () => {
    if (!sessionId) return;
    generateBtn.disabled = true;
    reportPlaceholder.hidden = true;
    reportSections.hidden = true;
    reportPlaceholder.hidden = false;
    reportPlaceholder.innerHTML = '<span class="spinner"></span>Generating report...';

    try {
      const res = await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Report generation failed");

      lastReportText = data.report;
      const { findings, impression } = parseReport(data.report);

      findingsText.textContent = findings;
      if (impression) {
        impressionText.textContent = impression;
        impressionHeading.hidden = false;
      } else {
        impressionHeading.hidden = true;
        impressionText.textContent = "";
      }
      reportModeNote.textContent =
        data.mode === "demo"
          ? "Mode: demo — template-based report, not a real diagnosis."
          : `Mode: ${data.mode}`;

      reportBox.textContent = data.report;
      rawReport.hidden = false;

      reportPlaceholder.hidden = true;
      reportSections.hidden = false;

      renderFindingTags(data.report);

      downloadBtn.disabled = false;

      chatWindow.innerHTML = "";
      addMessage("system", `Report ready (mode: ${data.mode}). Ask a question below.`);
      chatInput.disabled = false;
      chatSend.disabled = false;
    } catch (err) {
      reportPlaceholder.hidden = false;
      reportPlaceholder.textContent = "";
      reportSections.hidden = true;
      setError(err.message);
    } finally {
      generateBtn.disabled = false;
    }
  });

  function addMessage(role, text) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = text;
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  async function sendChat(questionOverride) {
    const question = (questionOverride || chatInput.value).trim();
    if (!question || !sessionId) return;
    addMessage("user", question);
    chatInput.value = "";
    chatSend.disabled = true;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Chat failed");
      addMessage("assistant", data.reply);
    } catch (err) {
      addMessage("system", `Error: ${err.message}`);
    } finally {
      chatSend.disabled = false;
    }
  }

  chatSend.addEventListener("click", () => sendChat());
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChat();
  });

  exampleQuestions.addEventListener("click", (e) => {
    const btn = e.target.closest(".chip-btn");
    if (!btn || chatInput.disabled) return;
    sendChat(btn.dataset.q);
  });

  // ---- Web search for a disease/finding ----
  function runManualSearch() {
    const term = searchInput.value.trim();
    if (!term) return;
    searchWeb(term);
  }
  searchBtn.addEventListener("click", runManualSearch);
  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runManualSearch();
  });

  // ---- Download report as a text file ----
  downloadBtn.addEventListener("click", () => {
    if (!lastReportText) return;
    const meta = lastMetadata || {};
    const lines = [
      "CXR Insight - Generated Report",
      "================================",
      `Patient ID: ${meta.patient_id || "-"}`,
      `View: ${meta.view || "-"}`,
      `Age: ${meta.age || "-"}`,
      `Gender: ${meta.gender || "-"}`,
      `Ethnicity: ${meta.ethnicity || "-"}`,
      "",
      lastReportText,
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cxr_report_${meta.patient_id || "report"}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });

  resetBtn.addEventListener("click", () => {
    sessionId = null;
    lastMetadata = null;
    lastReportText = null;
    fileInput.value = "";
    dropzone.hidden = false;
    imageFrame.hidden = true;
    patientList.hidden = true;
    patientPlaceholder.hidden = false;
    generateBtn.disabled = false;

    reportPlaceholder.hidden = false;
    reportPlaceholder.textContent = 'Upload an image and click "Generate Report" to see results here.';
    reportSections.hidden = true;
    rawReport.hidden = true;
    foundTagsBox.hidden = true;
    ruledoutTagsBox.hidden = true;
    downloadBtn.disabled = true;

    chatWindow.innerHTML = '<div class="msg system">Welcome! Ask me about the findings, impression, or specific details in the report once it\'s generated.</div>';
    chatInput.disabled = true;
    chatSend.disabled = true;
    setError("");
  });
})();
