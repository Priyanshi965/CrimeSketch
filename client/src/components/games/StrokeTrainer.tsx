/**
 * STROKE TRAINER
 * ──────────────
 * The face is broken into 5 segments (face shape → brows → eyes → nose → lips).
 * Only the CURRENT segment glows; the rest are dimmed.
 * User draws that segment, clicks "Score This Step", sees instant per-step
 * feedback, then advances to the next. Final score = average of all steps.
 *
 * Mechanic: structured, tutorial-style, educational sequence.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  FACE_PROFILES,
  FaceSegment,
  drawProfileFace,
  getCanvasPos,
  makeReferenceCanvas,
  pixelScore,
} from "./faceUtils";

const W = 380;
const H = 380;

const STEPS: { segment: FaceSegment; label: string; tip: string }[] = [
  { segment: "face",  label: "Face Outline", tip: "Draw the oval jaw + hairline. Keep it symmetric." },
  { segment: "brows", label: "Eyebrows",     tip: "Arch lightly above each eye. Don't press too hard." },
  { segment: "eyes",  label: "Eyes",         tip: "Draw both eye ovals + small pupils inside." },
  { segment: "nose",  label: "Nose",         tip: "Bridge lines down, then flare out the nostrils." },
  { segment: "lips",  label: "Lips",         tip: "Upper lip has an M-curve; lower lip is a soft arc." },
];

interface Props {
  onGameComplete: (score: number) => void;
}

export default function StrokeTrainer({ onGameComplete }: Props) {
  const tplRef  = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef<HTMLCanvasElement>(null);
  const isDrawRef = useRef(false);
  const lastRef   = useRef<{ x: number; y: number } | null>(null);

  const profile = FACE_PROFILES[0]; // canonical balanced face

  const [phase, setPhase]       = useState<"intro" | "drawing" | "done">("intro");
  const [stepIdx, setStepIdx]   = useState(0);
  const [stepScores, setStepScores] = useState<number[]>([]);
  const [lastScore, setLastScore]   = useState<number | null>(null);
  const [brushSize, setBrushSize]   = useState(2);

  const step = STEPS[stepIdx];

  // ── Draw template canvas (full face, current segment highlighted) ─────────
  const refreshTemplate = useCallback(() => {
    const tpl = tplRef.current; if (!tpl) return;
    const ctx = tpl.getContext("2d")!;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0a0f1a"; ctx.fillRect(0, 0, W, H);
    drawProfileFace(ctx, W / 2, H / 2, profile, 1.0, "#00d4ff", step.segment);
  }, [profile, step.segment]);

  useEffect(() => { refreshTemplate(); }, [refreshTemplate]);

  // ── Clear user drawing canvas ─────────────────────────────────────────────
  const clearDraw = useCallback(() => {
    const dc = drawRef.current; if (!dc) return;
    dc.getContext("2d")!.clearRect(0, 0, W, H);
    setLastScore(null);
  }, []);

  useEffect(() => { clearDraw(); }, [stepIdx, clearDraw]);

  // ── Drawing events ────────────────────────────────────────────────────────
  const onDown  = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    drawRef.current!.setPointerCapture(e.pointerId);
    isDrawRef.current = true;
    lastRef.current = getCanvasPos(drawRef.current!, e);
  }, []);

  const onMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawRef.current) return;
    const dc = drawRef.current!;
    const ctx = dc.getContext("2d")!;
    const pos = getCanvasPos(dc, e);
    const last = lastRef.current; if (!last) { lastRef.current = pos; return; }
    ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(pos.x, pos.y);
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = brushSize;
    ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.stroke();
    lastRef.current = pos;
  }, [brushSize]);

  const onUp = useCallback(() => { isDrawRef.current = false; lastRef.current = null; }, []);

  // ── Score current step ────────────────────────────────────────────────────
  const scoreStep = useCallback(() => {
    const dc = drawRef.current; if (!dc) return;
    const ref = makeReferenceCanvas(profile, W, H, step.segment);
    const { overall } = pixelScore(dc, ref);
    setLastScore(overall);
    const updated = [...stepScores, overall];
    setStepScores(updated);

    if (stepIdx >= STEPS.length - 1) {
      const avg = Math.round(updated.reduce((s, v) => s + v, 0) / updated.length);
      setTimeout(() => { setPhase("done"); onGameComplete(avg); }, 900);
    } else {
      setTimeout(() => setStepIdx(i => i + 1), 900);
    }
  }, [profile, step.segment, stepScores, stepIdx, onGameComplete]);

  const restart = () => { setPhase("intro"); setStepIdx(0); setStepScores([]); setLastScore(null); clearDraw(); };

  // ── Intro screen ──────────────────────────────────────────────────────────
  if (phase === "intro") return (
    <div className="sg-intro">
      <div className="sg-intro-icon">🖊</div>
      <h2>Stroke Trainer</h2>
      <p>Draw one facial region at a time.<br />
         The highlighted area glows — trace it as accurately as you can.<br />
         Score each step before advancing. 5 steps total.</p>
      <button className="fc-btn primary" onClick={() => setPhase("drawing")}>Begin Training</button>
    </div>
  );

  // ── Done screen ───────────────────────────────────────────────────────────
  if (phase === "done") {
    const avg = Math.round(stepScores.reduce((s, v) => s + v, 0) / stepScores.length);
    return (
      <div className="sg-intro">
        <div className="sg-intro-icon">🏁</div>
        <h2>Training Complete</h2>
        <div className="sg-step-results">
          {STEPS.map((s, i) => (
            <div key={s.segment} className="sg-step-result-row">
              <span>{s.label}</span>
              <div className="sg-mini-bar"><div style={{ width: `${stepScores[i] ?? 0}%`, background: stepScores[i] >= 70 ? "#00ffcc" : stepScores[i] >= 45 ? "#00d4ff" : "#6a9abf" }} /></div>
              <span className="sg-step-score">{stepScores[i] ?? 0}%</span>
            </div>
          ))}
        </div>
        <div className="sg-final-score">{avg}<small>/100</small></div>
        <button className="fc-btn primary" onClick={restart}>Play Again</button>
      </div>
    );
  }

  const pct = Math.round((stepIdx / STEPS.length) * 100);

  // ── Main drawing UI ───────────────────────────────────────────────────────
  return (
    <div className="sg-trainer-shell">
      {/* Progress bar */}
      <div className="sg-progress-bar">
        <div className="sg-progress-fill" style={{ width: `${pct}%` }} />
      </div>

      <div className="sg-trainer-body">
        {/* Left: step info */}
        <div className="sg-trainer-info">
          <div className="sg-step-counter">Step {stepIdx + 1} / {STEPS.length}</div>
          <div className="sg-step-name">{step.label}</div>
          <div className="sg-step-tip">{step.tip}</div>

          <div className="sg-step-list">
            {STEPS.map((s, i) => (
              <div key={s.segment} className={`sg-step-item ${i === stepIdx ? "active" : i < stepIdx ? "done" : ""}`}>
                <span className="sg-step-dot">{i < stepIdx ? "✓" : i + 1}</span>
                <span>{s.label}</span>
                {stepScores[i] !== undefined && <span className="sg-step-sc">{stepScores[i]}%</span>}
              </div>
            ))}
          </div>

          <div className="sg-tool-row" style={{ marginTop: 16 }}>
            <span>Brush</span>
            <input type="range" min={1} max={6} value={brushSize} onChange={e => setBrushSize(+e.target.value)} />
            <span>{brushSize}px</span>
          </div>
        </div>

        {/* Center: canvas */}
        <div className="sg-canvas-col">
          <div className="sg-canvas-label">Draw the <strong>{step.label}</strong></div>
          <div className="sg-canvas-stack" style={{ width: W, height: H }}>
            <canvas ref={tplRef}  width={W} height={H} className="sg-canvas-bg" />
            <canvas ref={drawRef} width={W} height={H} className="sg-canvas-fg"
              style={{ cursor: "crosshair" }}
              onPointerDown={onDown} onPointerMove={onMove}
              onPointerUp={onUp}    onPointerLeave={onUp} />
          </div>
          <div className="sg-canvas-actions">
            <button className="fc-btn primary" onClick={scoreStep}>Score This Step</button>
            <button className="fc-btn" onClick={clearDraw}>Clear</button>
          </div>
          {lastScore !== null && (
            <div className={`sg-step-feedback ${lastScore >= 70 ? "good" : lastScore >= 45 ? "ok" : "low"}`}>
              {step.label}: {lastScore}% — {lastScore >= 70 ? "Excellent!" : lastScore >= 45 ? "Good effort" : "Keep practicing"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
