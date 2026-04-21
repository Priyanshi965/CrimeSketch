/**
 * MATCH BOOST
 * ───────────
 * Investigator's sandbox. No templates, no timers, no constraints.
 *
 * The player draws a suspect face freely, then submits to the REAL
 * /predict API. After submission:
 *   • The animated pipeline shows each AI step lighting up in sequence
 *   • Sketch Quality % and AI Match Confidence % are shown side-by-side
 *   • Matched suspect cards are revealed one at a time
 *   • An insight note explains the quality↔confidence relationship
 *
 * Mechanic: real-world consequence — this is the actual system.
 *           Educational about how sketch quality affects AI accuracy.
 */

import { useCallback, useRef, useState } from "react";
import { trpc } from "@/lib/trpc";
import { toast } from "sonner";
import {
  FACE_PROFILES,
  drawProfileFace,
  getCanvasPos,
  makeReferenceCanvas,
  pixelScore,
} from "./faceUtils";

const W = 380;
const H = 380;

const PIPELINE = [
  { id: "upload",    label: "Upload",     icon: "📤" },
  { id: "preprocess",label: "Preprocess", icon: "⚙️" },
  { id: "embed",     label: "ResNet50",   icon: "🧠" },
  { id: "faiss",     label: "FAISS",      icon: "⚡" },
  { id: "rank",      label: "Rank",       icon: "📊" },
  { id: "result",    label: "Result",     icon: "✅" },
];

interface Props {
  onGameComplete: (score: number) => void;
}

export default function MatchBoost({ onGameComplete }: Props) {
  const drawRef   = useRef<HTMLCanvasElement>(null);
  const tplRef    = useRef<HTMLCanvasElement>(null); // dark bg only, no template
  const isDrawRef = useRef(false);
  const lastRef   = useRef<{ x: number; y: number } | null>(null);

  const [phase, setPhase]         = useState<"draw" | "loading" | "result">("draw");
  const [activePipe, setActivePipe] = useState(-1);
  const [quality, setQuality]     = useState(0);
  const [matches, setMatches]     = useState<any[]>([]);
  const [revealedCount, setRevealedCount] = useState(0);
  const [brushSize, setBrushSize] = useState(2);
  const [isEraser, setIsEraser]   = useState(false);

  const predictMutation = trpc.ml.predict.useMutation();

  const clearDraw = useCallback(() => {
    const dc = drawRef.current; if (!dc) return;
    dc.getContext("2d")!.clearRect(0, 0, W, H);
  }, []);

  // Init dark-background template canvas
  const initTpl = useCallback(() => {
    const tpl = tplRef.current; if (!tpl) return;
    const ctx = tpl.getContext("2d")!;
    ctx.fillStyle = "#0a0f1a"; ctx.fillRect(0, 0, W, H);
  }, []);

  // ── Drawing ───────────────────────────────────────────────────────────────
  const onDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (phase !== "draw") return;
    drawRef.current!.setPointerCapture(e.pointerId);
    isDrawRef.current = true;
    lastRef.current = getCanvasPos(drawRef.current!, e);
  }, [phase]);

  const onMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawRef.current || phase !== "draw") return;
    const dc = drawRef.current!;
    const ctx = dc.getContext("2d")!;
    const pos = getCanvasPos(dc, e);
    const last = lastRef.current; if (!last) { lastRef.current = pos; return; }
    ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(pos.x, pos.y);
    if (isEraser) {
      ctx.globalCompositeOperation = "destination-out";
      ctx.strokeStyle = "rgba(0,0,0,1)"; ctx.lineWidth = brushSize * 4;
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = brushSize;
    }
    ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.stroke();
    ctx.globalCompositeOperation = "source-over";
    lastRef.current = pos;
  }, [phase, brushSize, isEraser]);

  const onUp = useCallback(() => { isDrawRef.current = false; lastRef.current = null; }, []);

  // ── Animate pipeline ──────────────────────────────────────────────────────
  const animatePipeline = (onDone: () => void) => {
    let idx = 0;
    const id = setInterval(() => {
      setActivePipe(idx);
      idx++;
      if (idx >= PIPELINE.length) {
        clearInterval(id);
        setTimeout(onDone, 300);
      }
    }, 340);
  };

  // ── Submit sketch ─────────────────────────────────────────────────────────
  const submit = useCallback(async () => {
    const dc = drawRef.current; if (!dc) return;

    // Compute sketch quality against balanced face profile
    const ref = makeReferenceCanvas(FACE_PROFILES[0], W, H);
    const { overall } = pixelScore(dc, ref);
    setQuality(overall);

    // Composite: dark bg + user strokes
    const comp = document.createElement("canvas");
    comp.width = W; comp.height = H;
    const cctx = comp.getContext("2d")!;
    cctx.fillStyle = "#0a0f1a"; cctx.fillRect(0, 0, W, H);
    cctx.drawImage(dc, 0, 0);
    const base64 = comp.toDataURL("image/png").split(",")[1];

    setPhase("loading");
    setActivePipe(-1);
    setMatches([]);
    setRevealedCount(0);

    animatePipeline(async () => {
      try {
        const res = await predictMutation.mutateAsync({ imageData: base64, topK: 5 });
        const found = (res.matches || []) as any[];
        setMatches(found);
        // Reveal cards one at a time
        let i = 0;
        const reveal = setInterval(() => {
          i++;
          setRevealedCount(i);
          if (i >= found.length) clearInterval(reveal);
        }, 250);
        setPhase("result");
        onGameComplete(Math.min(100, overall + Math.round((found[0]?.confidence ?? 0) * 30)));
        toast.success("Sketch processed by AI!");
      } catch {
        toast.error("Match API offline — score recorded.");
        setPhase("result");
        onGameComplete(overall);
      }
    });
  }, [predictMutation, onGameComplete]);

  const reset = () => {
    setPhase("draw");
    setActivePipe(-1);
    setMatches([]);
    setRevealedCount(0);
    clearDraw();
    initTpl();
  };

  const qualityColor = quality >= 70 ? "#00ffcc" : quality >= 45 ? "#00d4ff" : "#ffd700";

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="sg-boost-shell">
      <div className="sg-boost-body">
        {/* Left: canvas */}
        <div className="sg-canvas-col">
          <div className="sg-boost-canvas-label">
            {phase === "draw"    && "Draw your suspect sketch"}
            {phase === "loading" && "Submitting to AI…"}
            {phase === "result"  && "Sketch submitted ✓"}
          </div>
          <div className="sg-canvas-stack" style={{ width: W, height: H }}>
            <canvas ref={tplRef}  width={W} height={H} className="sg-canvas-bg" />
            <canvas ref={drawRef} width={W} height={H} className="sg-canvas-fg"
              style={{ cursor: phase === "draw" ? (isEraser ? "cell" : "crosshair") : "default" }}
              onPointerDown={onDown} onPointerMove={onMove}
              onPointerUp={onUp}    onPointerLeave={onUp} />
          </div>
          {phase === "draw" && (
            <div className="sg-canvas-actions">
              <button className="fc-btn primary" onClick={submit}
                disabled={predictMutation.isPending}>Submit to AI Matcher</button>
              <button className="fc-btn" onClick={clearDraw}>Clear</button>
              <button className={`fc-btn ${isEraser ? "primary" : ""}`}
                onClick={() => setIsEraser(p => !p)}>
                {isEraser ? "Eraser" : "Brush"}
              </button>
              <div className="sg-tool-row">
                <span>Size</span>
                <input type="range" min={1} max={8} value={brushSize}
                  onChange={e => setBrushSize(+e.target.value)} />
                <span>{brushSize}px</span>
              </div>
            </div>
          )}
          {phase === "result" && (
            <div className="sg-canvas-actions">
              <button className="fc-btn primary" onClick={reset}>Draw Another</button>
            </div>
          )}
        </div>

        {/* Right: pipeline + results */}
        <div className="sg-boost-right">
          {/* AI Pipeline */}
          <div className="sg-boost-pipeline">
            <div className="fc-section-title">AI Pipeline</div>
            <div className="sg-pipeline-steps">
              {PIPELINE.map((step, i) => (
                <div key={step.id}
                  className={`sg-pipe-step ${i <= activePipe ? "active" : ""} ${i === activePipe ? "current" : ""}`}>
                  <span className="sg-pipe-icon">{step.icon}</span>
                  <span>{step.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Quality vs Confidence */}
          {phase === "result" && (
            <div className="sg-boost-metrics">
              <div className="sg-metric-card">
                <div>Sketch Quality</div>
                <div className="sg-metric-val" style={{ color: qualityColor }}>{quality}%</div>
              </div>
              <div className="sg-metric-sep">→</div>
              <div className="sg-metric-card">
                <div>Top Match</div>
                <div className="sg-metric-val" style={{ color: "#00d4ff" }}>
                  {matches[0] ? `${((matches[0].confidence ?? 0) * 100).toFixed(1)}%` : "—"}
                </div>
              </div>
            </div>
          )}

          {/* Match insight */}
          {phase === "result" && quality > 0 && (
            <div className="sg-boost-insight">
              {quality >= 65
                ? "✓ Strong sketch structure — AI could extract clear features."
                : quality >= 35
                ? "△ Partial sketch — AI matched on available features. Try adding more detail."
                : "✗ Low sketch coverage — Add face shape, eyes, and nose for better matches."}
            </div>
          )}

          {/* Match results */}
          {matches.length > 0 && (
            <div className="sg-boost-matches">
              <div className="fc-section-title" style={{ marginBottom: 8 }}>Top Matches</div>
              {matches.slice(0, revealedCount).map((m, i) => (
                <div key={i} className="sg-boost-match-row">
                  <span className="sg-match-rank">#{i + 1}</span>
                  {m.image_url
                    ? <img src={m.image_url} alt={m.name} className="sg-match-thumb" />
                    : <div className="sg-match-thumb sg-match-ph" />}
                  <div className="sg-match-info">
                    <strong>{m.name ?? "Unknown"}</strong>
                    <span>{m.city ?? "—"} · {m.crime_type ?? "—"}</span>
                  </div>
                  <div className="sg-match-conf" style={{ color: i === 0 ? "#00ffcc" : "#00d4ff" }}>
                    {((m.confidence ?? 0) * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
