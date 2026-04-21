// Shared face drawing utilities for all game modes.
// All faces are drawn via canvas 2D API — no images required.

export interface FaceProfile {
  id: number;
  label: string;
  faceRx: number;   // horizontal radius of face oval
  faceRy: number;   // vertical radius
  eyeOffX: number;  // horizontal distance of each eye from center
  eyeOffY: number;  // vertical offset of eyes from canvas center (negative = up)
  eyeRx: number;    // eye horizontal radius
  eyeRy: number;    // eye vertical radius
  browDY: number;   // brow offset above eye center (positive = higher)
  browW: number;    // brow half-width
  noseLen: number;  // nose bridge length
  noseW: number;    // nostril spread
  lipOffY: number;  // lip center offset below canvas center
  lipW: number;     // lip half-width
}

export const FACE_PROFILES: FaceProfile[] = [
  { id: 0, label: "Oval · Balanced",      faceRx: 110, faceRy: 152, eyeOffX: 50, eyeOffY: -52, eyeRx: 28, eyeRy: 13, browDY: 22, browW: 54, noseLen: 50, noseW: 16, lipOffY: 62, lipW: 40 },
  { id: 1, label: "Round · Wide Eyes",    faceRx: 132, faceRy: 138, eyeOffX: 56, eyeOffY: -44, eyeRx: 36, eyeRy: 17, browDY: 24, browW: 64, noseLen: 44, noseW: 19, lipOffY: 57, lipW: 47 },
  { id: 2, label: "Narrow · Deep-Set",   faceRx:  94, faceRy: 168, eyeOffX: 43, eyeOffY: -60, eyeRx: 22, eyeRy:  9, browDY: 18, browW: 44, noseLen: 58, noseW: 13, lipOffY: 70, lipW: 35 },
  { id: 3, label: "Square Jaw",           faceRx: 126, faceRy: 144, eyeOffX: 50, eyeOffY: -48, eyeRx: 26, eyeRy: 11, browDY: 16, browW: 52, noseLen: 52, noseW: 21, lipOffY: 66, lipW: 45 },
  { id: 4, label: "Heart · Large Eyes",  faceRx: 118, faceRy: 160, eyeOffX: 58, eyeOffY: -64, eyeRx: 33, eyeRy: 16, browDY: 26, browW: 62, noseLen: 46, noseW: 14, lipOffY: 54, lipW: 36 },
  { id: 5, label: "Long · Heavy Brows",  faceRx: 101, faceRy: 173, eyeOffX: 46, eyeOffY: -56, eyeRx: 29, eyeRy: 12, browDY: 20, browW: 57, noseLen: 60, noseW: 17, lipOffY: 74, lipW: 43 },
];

export type FaceSegment = "face" | "brows" | "eyes" | "nose" | "lips";

export function drawProfileFace(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  p: FaceProfile,
  alpha: number,
  baseColor: string,
  highlight?: FaceSegment, // if set, only this segment is bright
) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  const full = highlight === undefined;
  const dim = baseColor + "44";

  function seg(s: FaceSegment, w: number) {
    ctx.strokeStyle = full || highlight === s ? baseColor : dim;
    ctx.lineWidth    = full || highlight === s ? w : w * 0.7;
  }

  const ey = cy + p.eyeOffY;

  // ── Face oval ─────────────────────────────────────────────
  seg("face", 2.5);
  ctx.beginPath();
  ctx.ellipse(cx, cy + 8, p.faceRx, p.faceRy, 0, 0, Math.PI * 2);
  ctx.stroke();
  // Hairline arc
  ctx.beginPath();
  ctx.arc(cx, cy - p.faceRy + 30, p.faceRx + 14, Math.PI + 0.3, -0.3);
  ctx.stroke();

  // ── Eyebrows ───────────────────────────────────────────────
  seg("brows", 2.5);
  const bY = ey - p.browDY;
  ctx.beginPath();
  ctx.moveTo(cx - p.eyeOffX - p.browW + 8, bY + 6);
  ctx.quadraticCurveTo(cx - p.eyeOffX, bY - 5, cx - p.eyeOffX + p.browW - 8, bY + 5);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx + p.eyeOffX - p.browW + 8, bY + 5);
  ctx.quadraticCurveTo(cx + p.eyeOffX, bY - 5, cx + p.eyeOffX + p.browW - 8, bY + 6);
  ctx.stroke();

  // ── Eyes ───────────────────────────────────────────────────
  seg("eyes", 2);
  ctx.beginPath(); ctx.ellipse(cx - p.eyeOffX, ey, p.eyeRx, p.eyeRy, 0, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx - p.eyeOffX, ey, Math.max(4, p.eyeRy - 4), 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.ellipse(cx + p.eyeOffX, ey, p.eyeRx, p.eyeRy, 0, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx + p.eyeOffX, ey, Math.max(4, p.eyeRy - 4), 0, Math.PI * 2); ctx.stroke();

  // ── Nose ───────────────────────────────────────────────────
  seg("nose", 2);
  const ny = ey + p.noseLen;
  ctx.beginPath(); ctx.moveTo(cx, ey + 10); ctx.lineTo(cx - p.noseW * 0.5, ny); ctx.lineTo(cx - p.noseW, ny + 8); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, ey + 10); ctx.lineTo(cx + p.noseW * 0.5, ny); ctx.lineTo(cx + p.noseW, ny + 8); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx - p.noseW, ny + 12, p.noseW * 0.58, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx + p.noseW, ny + 12, p.noseW * 0.58, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx - p.noseW * 1.6, ny + 12); ctx.lineTo(cx + p.noseW * 1.6, ny + 12); ctx.stroke();

  // ── Lips ───────────────────────────────────────────────────
  seg("lips", 2);
  const ly = cy + p.lipOffY;
  ctx.beginPath();
  ctx.moveTo(cx - p.lipW, ly);
  ctx.quadraticCurveTo(cx - p.lipW * 0.4, ly - 8, cx - p.lipW * 0.1, ly - 3);
  ctx.quadraticCurveTo(cx, ly - 7, cx + p.lipW * 0.1, ly - 3);
  ctx.quadraticCurveTo(cx + p.lipW * 0.4, ly - 8, cx + p.lipW, ly);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx - p.lipW, ly);
  ctx.quadraticCurveTo(cx, ly + Math.round(p.lipW * 0.42), cx + p.lipW, ly);
  ctx.stroke();

  ctx.restore();
}

/** Render a face profile to a data URL (used by FaceQuiz thumbnails). */
export function profileToDataURL(p: FaceProfile, size: number, color = "#6a9abf"): string {
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#0a0f1a";
  ctx.fillRect(0, 0, size, size);
  // Scale profile from 400×400 coordinate space to `size`
  const scale = size / 400;
  const scaled: FaceProfile = {
    ...p,
    faceRx: p.faceRx * scale, faceRy: p.faceRy * scale,
    eyeOffX: p.eyeOffX * scale, eyeOffY: p.eyeOffY * scale,
    eyeRx: p.eyeRx * scale, eyeRy: p.eyeRy * scale,
    browDY: p.browDY * scale, browW: p.browW * scale,
    noseLen: p.noseLen * scale, noseW: p.noseW * scale,
    lipOffY: p.lipOffY * scale, lipW: p.lipW * scale,
  };
  drawProfileFace(ctx, size / 2, size / 2 + 8 * scale, scaled, 1.0, color);
  return c.toDataURL();
}

/** Canvas coordinate from pointer event, scaled to canvas logical size. */
export function getCanvasPos(
  canvas: HTMLCanvasElement,
  e: React.PointerEvent,
): { x: number; y: number } {
  const r = canvas.getBoundingClientRect();
  return {
    x: ((e.clientX - r.left) / r.width)  * canvas.width,
    y: ((e.clientY - r.top)  / r.height) * canvas.height,
  };
}

/** Simple pixel-overlap F1 score: user drawing vs reference (white-on-black). */
export function pixelScore(
  drawCanvas: HTMLCanvasElement,
  refCanvas: HTMLCanvasElement,
): { overall: number; recall: number; precision: number } {
  const W = drawCanvas.width, H = drawCanvas.height;
  const refD = refCanvas.getContext("2d")!.getImageData(0, 0, W, H).data;
  const usrD = drawCanvas.getContext("2d")!.getImageData(0, 0, W, H).data;
  let tPx = 0, uPx = 0, hit = 0;
  for (let i = 0; i < refD.length; i += 4) {
    const t = refD[i] > 80;
    const u = usrD[i] > 80 || usrD[i + 1] > 80 || usrD[i + 2] > 80;
    if (t) tPx++;
    if (u) uPx++;
    if (t && u) hit++;
  }
  if (!tPx) return { overall: 0, recall: 0, precision: 0 };
  const rec = hit / tPx;
  const prec = uPx > 0 ? hit / uPx : 0;
  const f1 = rec + prec > 0 ? (2 * rec * prec) / (rec + prec) : 0;
  return { overall: Math.round(f1 * 100), recall: Math.round(rec * 100), precision: Math.round(prec * 100) };
}

/** Build a white-on-black reference canvas from a profile. */
export function makeReferenceCanvas(
  p: FaceProfile,
  W: number,
  H: number,
  segment?: FaceSegment,
): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);
  drawProfileFace(ctx, W / 2, H / 2, p, 1.0, "#ffffff", segment);
  return c;
}
