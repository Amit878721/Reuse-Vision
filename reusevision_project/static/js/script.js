(() => {
  const input = document.getElementById('imageInput');
  const dropzone = document.getElementById('dropzone');
  const previewBody = document.getElementById('previewBody');
  const fileName = document.getElementById('fileName');
  const fileSize = document.getElementById('fileSize');
  const detectBtn = document.getElementById('detectBtn');
  const confidenceInput = document.getElementById('confidenceInput');
  const errorBox = document.getElementById('errorBox');
  const loading = document.getElementById('loading');
  const resultsPlaceholder = document.getElementById('resultsPlaceholder');
  const resultsPanel = document.getElementById('resultsPanel');
  const downloadExperimentBtn = document.getElementById('downloadExperimentBtn');
  let selectedFile = null;
  let previewUrl = null;
  let lastExperiment = null;

  const showError = (message) => {
    errorBox.textContent = message;
    errorBox.hidden = false;
  };
  const clearError = () => { errorBox.hidden = true; errorBox.textContent = ''; };
  const formatSize = bytes => bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(2)} MB`;

  function selectFile(file) {
    clearError();
    if (!file) return;
    const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!allowed.includes(file.type)) { showError('Unsupported image. Please use PNG, JPG, WEBP, or BMP.'); return; }
    if (file.size > 12 * 1024 * 1024) { showError('Image is too large. Maximum size is 12 MB.'); return; }
    selectedFile = file;
    lastExperiment = null;
    downloadExperimentBtn.disabled = true;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    previewBody.replaceChildren();
    const img = document.createElement('img');
    img.src = previewUrl;
    img.alt = 'Selected hardware image preview';
    previewBody.appendChild(img);
    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);
    detectBtn.disabled = false;
    detectBtn.querySelector('span').textContent = 'Run AI Detection';
  }

  input.addEventListener('change', () => selectFile(input.files[0]));
  ['dragenter', 'dragover'].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.remove('dragover'); }));
  dropzone.addEventListener('drop', event => selectFile(event.dataTransfer.files[0]));

  function renderResults(data) {
    document.getElementById('totalCount').textContent = data.total;
    document.getElementById('classCount').textContent = Object.keys(data.counts).length;
    document.getElementById('inferenceTime').textContent = `${data.inference_ms} ms`;
    document.getElementById('meanConfidence').textContent = `${data.mean_confidence}%`;
    document.getElementById('annotatedImage').src = data.annotated_image;
    document.getElementById('downloadBtn').href = data.download_url;

    const countsList = document.getElementById('countsList');
    countsList.replaceChildren();
    if (!Object.keys(data.counts).length) {
      const empty = document.createElement('div');
      empty.className = 'count-empty';
      empty.textContent = 'No supported components detected at this confidence threshold.';
      countsList.appendChild(empty);
    } else {
      Object.entries(data.counts).forEach(([name, count]) => {
        const row = document.createElement('div'); row.className = 'count-row';
        const label = document.createElement('span'); label.textContent = name;
        const value = document.createElement('strong'); value.textContent = count;
        row.append(label, value); countsList.appendChild(row);
      });
    }

    const detectionsList = document.getElementById('detectionsList');
    detectionsList.replaceChildren();
    if (!data.detections.length) {
      const empty = document.createElement('div');
      empty.className = 'count-empty';
      empty.textContent = 'No detections to list. Try a clearer motherboard photo or another threshold.';
      detectionsList.appendChild(empty);
    } else {
      data.detections.forEach((item, index) => {
        const row = document.createElement('div'); row.className = 'detection-item';
        const main = document.createElement('div'); main.className = 'detection-main';
        const name = document.createElement('span'); name.textContent = `${index + 1}. ${item.class}`;
        const confidence = document.createElement('span'); confidence.className = 'confidence'; confidence.textContent = `${item.confidence}%`;
        main.append(name, confidence);
        const coords = document.createElement('div'); coords.className = 'detection-coords'; coords.textContent = `Box: ${item.box.map(v => Math.round(v)).join(', ')}`;
        row.append(main, coords); detectionsList.appendChild(row);
      });
    }

    const experimentRows = document.getElementById('experimentRows');
    experimentRows.replaceChildren();
    data.threshold_experiment.forEach(item => {
      const row = document.createElement('tr');
      if (Math.abs(item.threshold - data.selected_confidence) < 0.0001) row.className = 'selected-threshold';
      const values = [item.threshold.toFixed(2), item.total, `${item.mean_confidence}%`, item.inference_ms];
      values.forEach(value => {
        const cell = document.createElement('td');
        cell.textContent = value;
        row.appendChild(cell);
      });
      experimentRows.appendChild(row);
    });
    document.getElementById('experimentNote').textContent = data.experiment_note;
    lastExperiment = {
      imageName: selectedFile ? selectedFile.name : 'uploaded_image',
      selectedConfidence: data.selected_confidence,
      rows: data.threshold_experiment
    };
    downloadExperimentBtn.disabled = false;
    resultsPlaceholder.hidden = true;
    resultsPanel.hidden = false;
  }

  detectBtn.addEventListener('click', async () => {
    if (!selectedFile) { showError('Please choose an image first.'); return; }
    clearError(); loading.hidden = false; detectBtn.disabled = true;
    detectBtn.querySelector('span').textContent = 'Running experiment…';
    const form = new FormData();
    form.append('image', selectedFile);
    form.append('confidence', confidenceInput.value);
    try {
      const response = await fetch('/detect', { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Detection failed.');
      renderResults(data);
    } catch (error) {
      showError(error.message || 'Could not connect to the local detection server. Is Flask running?');
    } finally {
      loading.hidden = true;
      detectBtn.disabled = !selectedFile;
      detectBtn.querySelector('span').textContent = 'Run AI Detection';
    }
  });

  function csvEscape(value) {
    const text = String(value ?? '');
    return `"${text.replaceAll('"', '""')}"`;
  }

  downloadExperimentBtn.addEventListener('click', () => {
    if (!lastExperiment) return;
    const lines = [
      ['image', 'selected_threshold', 'threshold', 'detections', 'mean_detection_confidence_percent', 'inference_time_ms'].map(csvEscape).join(','),
      ...lastExperiment.rows.map(row => [
        lastExperiment.imageName,
        lastExperiment.selectedConfidence,
        row.threshold,
        row.total,
        row.mean_confidence,
        row.inference_ms
      ].map(csvEscape).join(','))
    ];
    const blob = new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'reusevision_threshold_experiment.csv';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  });
})();
