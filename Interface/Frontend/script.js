document.addEventListener("DOMContentLoaded", () => {

  /***********************
   * PAGE NAVIGATION
   ***********************/
  window.goHome = function () {
    hideAll();
    document.getElementById("home").classList.remove("hidden");
  };

  window.showDraw = function () {
    hideAll();
    document.getElementById("draw-area").classList.remove("hidden");
  };

  window.showUpload = function () {
    hideAll();
    document.getElementById("upload-area").classList.remove("hidden");
  };

  function hideAll() {
    document.getElementById("home").classList.add("hidden");
    document.getElementById("draw-area").classList.add("hidden");
    document.getElementById("upload-area").classList.add("hidden");
    document.getElementById("result").classList.add("hidden");
  }

  /***********************
   * CANVAS SETUP
   ***********************/
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "white";
  ctx.lineWidth = 8;
  ctx.lineCap = "round";

  let drawing = false;

  canvas.addEventListener("mousedown", (e) => {
    drawing = true;
    ctx.beginPath();

    const rect = canvas.getBoundingClientRect();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;

    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
  });

  canvas.addEventListener("mouseup", () => {
    drawing = false;
    ctx.beginPath();
  });

  canvas.addEventListener("mouseleave", () => {
    drawing = false;
    ctx.beginPath();
  });

  window.clearCanvas = function () {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById("result").classList.add("hidden");
  };

  /***********************
   * CANVAS → BACKEND
   ***********************/
  window.captureCanvas = function () {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;

    let minX = canvas.width, minY = canvas.height;
    let maxX = 0, maxY = 0;

    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const i = (y * canvas.width + x) * 4;
            if (data[i] > 10) {
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }
    }

    if (minX >= maxX || minY >= maxY) {
        alert("Draw something first");
        return;
    }

    const padding = 20;
    minX = Math.max(minX - padding, 0);
    minY = Math.max(minY - padding, 0);
    maxX = Math.min(maxX + padding, canvas.width);
    maxY = Math.min(maxY + padding, canvas.height);

    const w = maxX - minX;
    const h = maxY - minY;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 32;
    tempCanvas.height = 32;
    const tctx = tempCanvas.getContext("2d");

    // WHITE background (model expects this)
    tctx.fillStyle = "white";
    tctx.fillRect(0, 0, 32, 32);

    const scale = Math.min(32 / w, 32 / h);
    const dw = w * scale;
    const dh = h * scale;
    const dx = (32 - dw) / 2;
    const dy = (32 - dh) / 2;

    tctx.drawImage(canvas, minX, minY, w, h, dx, dy, dw, dh);

    // ✅ EXPLICIT INVERSION (SAFE)
    const smallImg = tctx.getImageData(0, 0, 32, 32);
    const d = smallImg.data;
    for (let i = 0; i < d.length; i += 4) {
        d[i] = 255 - d[i];
        d[i + 1] = 255 - d[i + 1];
        d[i + 2] = 255 - d[i + 2];
    }
    tctx.putImageData(smallImg, 0, 0);

    // ✅ NOW send to backend
    tempCanvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "canvas.png");

        try {
            const res = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                body: formData
            });

            const result = await res.json();
            showResult(result.character, result.confidence);
        } catch (err) {
            console.error(err);
            alert("Prediction failed");
        }
    });
};

  /***********************
   * UPLOAD IMAGE
   ***********************/
  window.previewImage = function () {
    const file = document.getElementById("fileInput").files[0];
    if (!file) return;

    document.getElementById("preview").src =
      URL.createObjectURL(file);
  };

  window.predictUpload = async function () {
    const file = document.getElementById("fileInput").files[0];
    if (!file) {
      alert("Please select an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    await sendToBackend(formData);
  };

  /***********************
   * FASTAPI COMMUNICATION
   ***********************/
  async function sendToBackend(formData) {
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
      });

      if (!response.ok) throw new Error("Backend error");

      const result = await response.json();
      showResult(result.character, result.confidence);

    } catch (err) {
      alert("Prediction failed. Is backend running?");
      console.error(err);
    }
  }

  /***********************
   * DISPLAY RESULT
   ***********************/
  function showResult(character, confidence) {
    document.getElementById("predictedChar").innerText = character;
    document.getElementById("confidence").innerText =
      `Confidence: ${confidence.toFixed(2)}%`;

    document.getElementById("result").classList.remove("hidden");
  }

});
