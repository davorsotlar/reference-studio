// Reference Studio face scan. No face images or landmarks leave this browser.
const PACKAGE_ROOT = new URL("./vendor/mediapipe/", import.meta.url);
const FRAME_INTERVAL_MS = 135;
const GREEN_AFTER_MS = 440;
const CAPTURE_AFTER_MS = 1250;
let sharedDetector = null;
let detectorPromise = null;

async function loadDetector() {
  if (sharedDetector) return sharedDetector;
  if (!detectorPromise) {
    detectorPromise = (async () => {
      const { FaceDetector, FilesetResolver } = await import(
        "./vendor/mediapipe/vision_bundle.mjs"
      );
      const vision = await FilesetResolver.forVisionTasks(
        new URL("./wasm", PACKAGE_ROOT).href
      );
      return FaceDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: new URL(
            "./model/blaze_face_short_range.tflite",
            PACKAGE_ROOT
          ).href,
          delegate: "CPU"
        },
        runningMode: "VIDEO",
        minDetectionConfidence: 0.57
      });
    })().then((detector) => {
      sharedDetector = detector;
      return detector;
    }).catch((error) => {
      detectorPromise = null;
      throw error;
    });
  }
  return detectorPromise;
}

export function projectFace(detection, videoWidth, videoHeight, frameWidth, frameHeight, mirrored = false) {
  const box = detection && detection.boundingBox;
  if (!box || !videoWidth || !videoHeight || !frameWidth || !frameHeight) return null;
  const scale = Math.max(frameWidth / videoWidth, frameHeight / videoHeight);
  const offsetX = (frameWidth - videoWidth * scale) / 2;
  const offsetY = (frameHeight - videoHeight * scale) / 2;
  const map = (x, y) => ({
    x: mirrored ? frameWidth - (x * scale + offsetX) : x * scale + offsetX,
    y: y * scale + offsetY
  });
  const p0 = map(box.originX, box.originY);
  const p1 = map(box.originX + box.width, box.originY + box.height);
  const center = map(box.originX + box.width / 2, box.originY + box.height / 2);
  const keypoints = (detection.keypoints || []).map((point) => (
    map(point.x * videoWidth, point.y * videoHeight)
  ));
  return {
    x: Math.min(p0.x, p1.x),
    y: Math.min(p0.y, p1.y),
    width: Math.abs(p1.x - p0.x),
    height: Math.abs(p1.y - p0.y),
    cx: center.x / frameWidth,
    cy: center.y / frameHeight,
    relativeHeight: Math.abs(p1.y - p0.y) / frameHeight,
    keypoints
  };
}

export function inspectFace(detections, videoWidth, videoHeight, frameWidth, frameHeight, kind, mirrored) {
  if (!detections || detections.length === 0) {
    return { good: false, reason: "Tražim lice u kadru", face: null };
  }
  if (detections.length !== 1) {
    return { good: false, reason: "U kadru neka bude samo jedna osoba", face: null };
  }
  const detection = detections[0];
  const face = projectFace(detection, videoWidth, videoHeight, frameWidth, frameHeight, mirrored);
  if (!face) return { good: false, reason: "Lice nije očitano", face: null };
  if ((detection.categories?.[0]?.score || 0) < 0.57) {
    return { good: false, reason: "Poboljšaj svjetlo na licu", face };
  }
  const full = kind === "full";
  const centerX = 0.5, centerY = full ? 0.34 : 0.46;
  const dx = Math.abs(face.cx - centerX), dy = Math.abs(face.cy - centerY);
  if (dx > (full ? 0.25 : 0.22) || dy > (full ? 0.32 : 0.25)) {
    return { good: false, reason: "Pomakni lice prema sredini oznaka", face };
  }
  if (face.relativeHeight < (full ? 0.07 : 0.16)) {
    return { good: false, reason: "Približi se kameri", face };
  }
  if (face.relativeHeight > (full ? 0.39 : 0.69)) {
    return { good: false, reason: "Odmakni se malo od kamere", face };
  }
  return { good: true, reason: "Lice poravnato — ostani mirno", face };
}

export function makeFaceScanner({ video, preview, canvas, kind, mirrored, onUpdate, onCapture, onError }) {
  let detector = null;
  let active = false;
  let animationId = 0;
  let lastScanAt = -Infinity;
  let lastVideoTime = -1;
  let anchor = null;
  let steadySince = null;
  let captured = false;
  let startupToken = 0;
  const ctx = canvas.getContext("2d");
  const update = typeof onUpdate === "function" ? onUpdate : () => {};

  function clear() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function draw(face, color) {
    const width = preview.clientWidth;
    const height = preview.clientHeight;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    const c = canvas.getContext("2d");
    c.setTransform(ratio, 0, 0, ratio, 0, 0);
    c.clearRect(0, 0, width, height);
    if (!face) return;

    const left = Math.max(5, face.x), top = Math.max(5, face.y);
    const right = Math.min(width - 5, face.x + face.width);
    const bottom = Math.min(height - 5, face.y + face.height);
    const x = left - 9, y = top - 12;
    const w = right - left + 18, h = bottom - top + 24;
    if (w <= 0 || h <= 0) return;

    c.strokeStyle = color;
    c.shadowColor = color;
    c.shadowBlur = 12;
    c.lineWidth = 1.7;
    const radius = Math.min(w, h) * 0.24;
    c.beginPath();
    c.roundRect(x, y, w, h, radius);
    c.stroke();
    c.shadowBlur = 0;

    const pts = face.keypoints.filter(p => p.x >= 0 && p.x <= width && p.y >= 0 && p.y <= height);
    c.fillStyle = color;
    c.strokeStyle = color;
    c.lineWidth = 0.9;
    const connections = [[0, 1], [0, 2], [1, 2], [2, 3], [0, 4], [1, 5]];
    c.globalAlpha = 0.44;
    for (const [a, b] of connections) {
      if (!pts[a] || !pts[b]) continue;
      c.beginPath();
      c.moveTo(pts[a].x, pts[a].y);
      c.lineTo(pts[b].x, pts[b].y);
      c.stroke();
    }
    c.globalAlpha = 1;
    for (const point of pts) {
      c.beginPath();
      c.arc(point.x, point.y, 2.5, 0, Math.PI * 2);
      c.fill();
    }
    const guideLines = 6;
    c.globalAlpha = 0.16;
    for (let i = 1; i < guideLines; i++) {
      const yy = y + (h * i / guideLines);
      c.beginPath();
      c.moveTo(x + 5, yy);
      c.lineTo(x + w - 5, yy);
      c.stroke();
    }
    c.globalAlpha = 1;
  }

  function stop() {
    active = false;
    startupToken++;
    if (animationId) cancelAnimationFrame(animationId);
    animationId = 0;
    steadySince = null;
    anchor = null;
    lastVideoTime = -1;
    captured = false;
    clear();
  }

  function reportFailure(error) {
    if (!active) return;
    stop();
    onError?.(error);
  }

  function tick(now) {
    if (!active || captured) return;
    animationId = requestAnimationFrame(tick);
    if (now - lastScanAt < FRAME_INTERVAL_MS) return;
    if (!video.srcObject || video.videoWidth === 0 || video.readyState < 2) {
      update({ phase: "searching", label: "Čekam kameru", progress: 0 });
      return;
    }
    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;
    lastScanAt = now;
    try {
      const result = detector.detectForVideo(video, now);
      const quality = inspectFace(
        result.detections, video.videoWidth, video.videoHeight,
        preview.clientWidth, preview.clientHeight, kind(), mirrored()
      );
      if (!quality.good) {
        steadySince = null;
        anchor = null;
        draw(quality.face, "#6feee2");
        update({ phase: "searching", label: quality.reason, progress: 0 });
        return;
      }

      const candidate = quality.face;
      if (!anchor || Math.abs(candidate.cx - anchor.cx) > 0.024 ||
        Math.abs(candidate.cy - anchor.cy) > 0.028 ||
        Math.abs(candidate.relativeHeight - anchor.relativeHeight) > 0.04) {
        anchor = { cx: candidate.cx, cy: candidate.cy, relativeHeight: candidate.relativeHeight };
        steadySince = now;
      }
      const duration = now - steadySince;
      const green = duration >= GREEN_AFTER_MS;
      draw(candidate, green ? "#a5ffb2" : "#71f7f1");
      update({
        phase: green ? "hold" : "tracking",
        label: green ? "ZELENO — NE MIČI SE" : "Očitavam lice — miruj",
        progress: green ? Math.min(1, (duration - GREEN_AFTER_MS) / (CAPTURE_AFTER_MS - GREEN_AFTER_MS)) : Math.min(0.18, duration / GREEN_AFTER_MS * 0.18)
      });
      if (duration >= CAPTURE_AFTER_MS) {
        captured = true;
        update({ phase: "captured", label: "Fotografiram automatski", progress: 1 });
        onCapture?.();
      }
    } catch (error) {
      reportFailure(error);
    }
  }

  async function start() {
    if (active) stop();
    active = true;
    const token = ++startupToken;
    captured = false;
    lastScanAt = -Infinity;
    lastVideoTime = -1;
    update({ phase: "loading", label: "Učitavam prepoznavanje lica…", progress: 0 });
    try {
      detector = await loadDetector();
      if (!active || token !== startupToken) return;
      update({ phase: "searching", label: "Potraži lice u okviru", progress: 0 });
      animationId = requestAnimationFrame(tick);
    } catch (error) {
      reportFailure(error);
    }
  }

  return { start, stop, isActive: () => active };
}
