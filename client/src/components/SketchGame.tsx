import { useCallback, useEffect, useRef, useState } from "react";
import { trpc } from "@/lib/trpc";
import { toast } from "sonner";

// ─── Types ────────────────────────────────────────────────────────────────────

type GameMode = "trace" | "ghost" | "focus" | "boost";
type FocusFeature = "full" | "outline" | "eyes" | "nose" | "lips";
type GamePhase = "idle" | "showing" | "drawing" | "result";

interface GameStats {
  xp: number;
  gamesPlayed: number;
  bestScore: number;
  history: { score: number; mode: string; date: string; xpGained: number }[];
}

interface ScoreResult {
  overall: number;
  precision: number;
  recall: number;
  label: string;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const W = 400;
const H = 400;
const STATS_KEY = "crimesketch_game_stats";

const LEVELS = [
  { minXp: 0,    label: "Beginner Artist",  color: "#6a9abf" },
  { minXp: 200,  label: "Sketch Analyst",   color: "#00d4ff" },
  { minXp: 500,  label: "Forensic Expert",  color: "#00ffcc" },
  { minXp: 1000, label: "CID Officer",      color: "#ffd700" },
];

const FEATURES: { id: FocusFeature; label: string }[] = [
  { id: "full",    label: "Full Face" },
  { id: "outline", label: "Face Shape" },
  { id: "eyes",    label: "Eyes" },
  { id: "nose",    label: "Nose" },
  { id: "lips",    label: "Lips" },
];

// ─── Persistence ──────────────────────────────────────────────────────────────

function loadStats(): GameStats {
  try {
    const raw = localStorage.getItem(STATS_KEY);
    if (raw) return JSON.parse(raw) as GameStats;
  } catch {}
  return { xp: 0, gamesPlayed: 0, bestScore: 0, history: [] };
}

function saveStats(s: GameStats) {
  localStorage.setItem(STATS_KEY, JSON.stringify(s));
}

function getLevel(xp: number) {
  let lv = LEVELS[0];
  for (const l of LEVELS) if (xp >= l.minXp) lv = l;
  return lv;
}

function xpProgress(xp: number) {
  let idx = 0;
  for (let i = 0; i < LEVELS.length; i++) if (xp >= LEVELS[i].minXp) idx = i;
  const cur = LEVELS[idx].minXp;
  const nxt = idx < LEVELS.length - 1 ? LEVELS[idx + 1].minXp : cur + 500;
  const pct = Math.min(100, Math.round(((xp - cur) / (nxt - cur)) * 100));
  return { pct, next: nxt };
}

// ─── Canvas Template ──────────────────────────────────────────────────────────

function drawFaceTemplate(
  ctx: CanvasRenderingContext2D,
  feature: FocusFeature,
  alpha: number,
  color = "#00d4ff",
) {
  if (alpha <= 0) return;
  const cx = W / 2, cy = H / 2;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (feature === "full" || feature === "outline") {
    // Face oval
    ctx.beginPath();
    ctx.ellipse(cx, cy + 10, 120, 155, 0, 0, Math.PI * 2);
    ctx.stroke();
    // Hairline
    ctx.beginPath();
    ctx.arc(cx, cy - 110, 127, Math.PI + 0.25, -0.25);
    ctx.stroke();
    // Left ear
    ctx.beginPath();
    ctx.arc(cx - 122, cy + 15, 11, 1.2, 4.5);
    ctx.stroke();
    // Right ear
    ctx.beginPath();
    ctx.arc(cx + 122, cy + 15, 11, -1.2, 1.2);
    ctx.stroke();
  }

  if (feature === "full" || feature === "eyes") {
    // Left eye
    ctx.beginPath();
    ctx.ellipse(cx - 52, cy - 50, 30, 14, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx - 52, cy - 50, 7, 0, Math.PI * 2);
    ctx.stroke();
    // Left eyebrow
    ctx.beginPath();
    ctx.moveTo(cx - 84, cy - 74);
    ctx.quadraticCurveTo(cx - 52, cy - 88, cx - 20, cy - 74);
    ctx.stroke();
    // Right eye
    ctx.beginPath();
    ctx.ellipse(cx + 52, cy - 50, 30, 14, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx + 52, cy - 50, 7, 0, Math.PI * 2);
    ctx.stroke();
    // Right eyebrow
    ctx.beginPath();
    ctx.moveTo(cx + 20, cy - 74);
    ctx.quadraticCurveTo(cx + 52, cy - 88, cx + 84, cy - 74);
    ctx.stroke();
  }

  if (feature === "full" || feature === "nose") {
    // Nose bridge + wings
    ctx.beginPath();
    ctx.moveTo(cx, cy - 38);
    ctx.lineTo(cx - 8, cy + 12);
    ctx.lineTo(cx - 20, cy + 26);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx, cy - 38);
    ctx.lineTo(cx + 8, cy + 12);
    ctx.lineTo(cx + 20, cy + 26);
    ctx.stroke();
    // Nostrils
    ctx.beginPath();
    ctx.arc(cx - 16, cy + 30, 10, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx + 16, cy + 30, 10, 0, Math.PI * 2);
    ctx.stroke();
    // Base line
    ctx.beginPath();
    ctx.moveTo(cx - 28, cy + 30);
    ctx.lineTo(cx + 28, cy + 30);
    ctx.stroke();
  }

  if (feature === "full" || feature === "lips") {
    // Upper lip (M shape)
    ctx.beginPath();
    ctx.moveTo(cx - 48, cy + 60);
    ctx.quadraticCurveTo(cx - 22, cy + 52, cx - 8, cy + 57);
    ctx.quadraticCurveTo(cx, cy + 53, cx + 8, cy + 57);
    ctx.quadraticCurveTo(cx + 22, cy + 52, cx + 48, cy + 60);
    ctx.stroke();
    // Lower lip
    ctx.beginPath();
    ctx.moveTo(cx - 48, cy + 60);
    ctx.quadraticCurveTo(cx, cy + 84, cx + 48, cy + 60);
    ctx.stroke();
    // Philtrum
    ctx.beginPath();
    ctx.moveTo(cx - 10, cy + 44);
    ctx.lineTo(cx - 8, cy + 57);
    ctx.moveTo(cx + 10, cy + 44);
    ctx.lineTo(cx + 8, cy + 57);
    ctx.stroke();
  }

  ctx.restore();
}

// ─── Scoring ──────────────────────────────────────────────────────────────────

function calculateScore(drawCanvas: HTMLCanvasElement, feature: FocusFeature): ScoreResult {
  const ref = document.createElement("canvas");
  ref.width = W;
  ref.height = H;
  const rctx = ref.getContext("2d")!;
  rctx.fillStyle = "#000000";
  rctx.fillRect(0, 0, W, H);
  drawFaceTemplate(rctx, feature, 1.0, "#ffffff");

  const refD = rctx.getImageData(0, 0, W, H).data;
  const usrD = drawCanvas.getContext("2d")!.getImageData(0, 0, W, H).data;

  let tPx = 0, uPx = 0, hit = 0;
  for (let i = 0; i < refD.length; i += 4) {
    const t = refD[i] > 80;
    const u = usrD[i] > 80 || usrD[i + 1] > 80 || usrD[i + 2] > 80;
    if (t) tPx++;
    if (u) uPx++;
    if (t && u) hit++;
  }

  if (!tPx) return { overall: 0, precision: 0, recall: 0, label: "No reference" };
  const recall = hit / tPx;
  const precision = uPx > 0 ? hit / uPx : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const overall = Math.round(f1 * 100);
  const label =
    overall >= 85 ? "Excellent" :
    overall >= 65 ? "Good" :
    overall >= 45 ? "Fair" :
    "Keep Practicing";
  return { overall, precision: Math.round(precision * 100), recall: Math.round(recall * 100), label };
}

// ─── Pointer helpers ──────────────────────────────────────────────────────────

function getPos(canvas: HTMLCanvasElement, e: React.PointerEvent): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((e.clientX - rect.left) / rect.width) * canvas.width,
    y: ((e.clientY - rect.top) / rect.height) * canvas.height,
  };
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function SketchGame() {
  const tplRef = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef<HTMLCanvasElement>(null);
  const isDrawingRef = useRef(false);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Use refs so timer callbacks always read current values
  const modeRef = useRef<GameMode>("trace");
  const featureRef = useRef<FocusFeature>("full");

  const [mode, setMode] = useState<GameMode>("trace");
  const [feature, setFeature] = useState<FocusFeature>("full");
  const [phase, setPhase] = useState<GamePhase>("idle");
  const [ghostCountdown, setGhostCountdown] = useState(4);
  const [scoreResult, setScoreResult] = useState<ScoreResult | null>(null);
  const [matchResult, setMatchResult] = useState<any>(null);
  const [stats, setStats] = useState<GameStats>(loadStats);
  const [brushSize, setBrushSize] = useState(2);
  const [isEraser, setIsEraser] = useState(false);
  const [xpGained, setXpGained] = useState(0);

  const predictMutation = trpc.ml.predict.useMutation();

  // Keep refs in sync with state
  useEffect(() => { modeRef.current = mode; }, [mode]);
  useEffect(() => { featureRef.current = feature; }, [feature]);

  const lv = getLevel(stats.xp);
  const prog = xpProgress(stats.xp);

  // Draws (or clears) the template canvas
  const refreshTemplate = useCallback((alpha: number) => {
    const tpl = tplRef.current;
    if (!tpl) return;
    const ctx = tpl.getContext("2d")!;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0a0f1a";
    ctx.fillRect(0, 0, W, H);
    const feat = modeRef.current === "focus" ? featureRef.current : "full";
    drawFaceTemplate(ctx, feat, alpha);
  }, []);

  const clearDraw = useCallback(() => {
    const dc = drawRef.current;
    if (!dc) return;
    const ctx = dc.getContext("2d")!;
    ctx.clearRect(0, 0, W, H);
  }, []);

  // Init canvases on mount
  useEffect(() => {
    refreshTemplate(0);
    clearDraw();
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [refreshTemplate, clearDraw]);

  const resetGame = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    setPhase("idle");
    setScoreResult(null);
    setMatchResult(null);
    setXpGained(0);
    setIsEraser(false);
    clearDraw();
    refreshTemplate(0);
  }, [clearDraw, refreshTemplate]);

  // Reset when mode/feature changes
  useEffect(() => { resetGame(); }, [mode, feature]); // eslint-disable-line react-hooks/exhaustive-deps

  const startGame = useCallback(() => {
    clearDraw();
    setScoreResult(null);
    setMatchResult(null);
    setXpGained(0);

    if (mode === "trace" || mode === "focus") {
      refreshTemplate(0.18);
      setPhase("drawing");
      toast.info(mode === "trace" ? "Trace the guide lines!" : "Focus on the highlighted feature.");
    } else if (mode === "ghost") {
      refreshTemplate(1.0);
      setPhase("showing");
      setGhostCountdown(4);

      let remaining = 4;
      timerRef.current = setInterval(() => {
        remaining--;
        setGhostCountdown(remaining);
        if (remaining <= 0) {
          if (timerRef.current) clearInterval(timerRef.current);
          // Fade out
          let alpha = 1.0;
          timerRef.current = setInterval(() => {
            alpha = Math.max(0, alpha - 0.1);
            refreshTemplate(alpha);
            if (alpha <= 0) {
              if (timerRef.current) clearInterval(timerRef.current);
              timerRef.current = null;
              setPhase("drawing");
              toast.info("Now draw from memory!");
            }
          }, 80);
        }
      }, 1000);
    } else {
      // boost
      refreshTemplate(0);
      setPhase("drawing");
      toast.info("Draw a face sketch then submit to match!");
    }
  }, [mode, clearDraw, refreshTemplate]);

  // ─── Drawing events ───────────────────────────────────────────────────────

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (phase !== "drawing") return;
    drawRef.current!.setPointerCapture(e.pointerId);
    isDrawingRef.current = true;
    lastPosRef.current = getPos(drawRef.current!, e);
  }, [phase]);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current || phase !== "drawing") return;
    const dc = drawRef.current!;
    const ctx = dc.getContext("2d")!;
    const pos = getPos(dc, e);
    const last = lastPosRef.current;
    if (!last) { lastPosRef.current = pos; return; }

    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (isEraser) {
      ctx.globalCompositeOperation = "destination-out";
      ctx.lineWidth = brushSize * 4;
      ctx.strokeStyle = "rgba(0,0,0,1)";
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.lineWidth = brushSize;
      ctx.strokeStyle = "#ffffff";
    }
    ctx.stroke();
    ctx.globalCompositeOperation = "source-over";
    lastPosRef.current = pos;
  }, [phase, brushSize, isEraser]);

  const onPointerUp = useCallback(() => {
    isDrawingRef.current = false;
    lastPosRef.current = null;
  }, []);

  // ─── Submit ───────────────────────────────────────────────────────────────

  const submitSketch = useCallback(async () => {
    const dc = drawRef.current;
    if (!dc) return;

    const feat = mode === "focus" ? feature : "full";
    const score = calculateScore(dc, feat);
    setScoreResult(score);

    const gained = Math.max(5, Math.round(score.overall * 2));
    setXpGained(gained);

    const newStats: GameStats = {
      xp: stats.xp + gained,
      gamesPlayed: stats.gamesPlayed + 1,
      bestScore: Math.max(stats.bestScore, score.overall),
      history: [
        { score: score.overall, mode, date: new Date().toLocaleDateString(), xpGained: gained },
        ...stats.history.slice(0, 9),
      ],
    };
    setStats(newStats);
    saveStats(newStats);

    if (mode === "boost") {
      // Composite: template bg + user strokes → send to predict API
      const comp = document.createElement("canvas");
      comp.width = W; comp.height = H;
      const cctx = comp.getContext("2d")!;
      cctx.drawImage(tplRef.current!, 0, 0);  // dark bg
      cctx.drawImage(dc, 0, 0);               // user strokes
      const base64 = comp.toDataURL("image/png").split(",")[1];
      try {
        const res = await predictMutation.mutateAsync({ imageData: base64, topK: 3 });
        setMatchResult(res);
        toast.success("Sketch sent to AI matcher!");
      } catch {
        toast.error("Match API unavailable — score recorded.");
      }
    }

    setPhase("result");
  }, [mode, feature, stats, predictMutation]);

  const leaderboard = [...stats.history].sort((a, b) => b.score - a.score).slice(0, 5);

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="fc-game-shell">
      {/* Mode tabs */}
      <div className="fc-game-tabs">
        {(["trace", "ghost", "focus", "boost"] as GameMode[]).map(m => (
          <button
            key={m}
            className={`fc-game-tab ${mode === m ? "active" : ""}`}
            onClick={() => setMode(m)}
          >
            {m === "trace" ? "Trace Mode" :
             m === "ghost" ? "Ghost Mode" :
             m === "focus" ? "Feature Focus" : "Match Boost"}
          </button>
        ))}
      </div>

      <div className="fc-game-body">
        {/* ── Left: info & controls ─────────────────────────────── */}
        <div className="fc-game-info">
          <div className="fc-section-title">Instructions</div>
          <p className="fc-game-desc">
            {mode === "trace" && "A faint face guide appears on the canvas. Trace over it to practice forensic face proportions and structure."}
            {mode === "ghost" && "Study the face template for 4 seconds — it then fades away. Redraw it entirely from memory."}
            {mode === "focus" && "Choose a facial feature below. A focused guide appears for that feature only. Perfect it before moving on."}
            {mode === "boost" && "Draw a suspect face sketch freely. Submit to send it through the real AI matcher and see confidence scores."}
          </p>

          {mode === "focus" && (
            <div className="fc-game-features">
              {FEATURES.map(f => (
                <button
                  key={f.id}
                  className={`fc-feature-btn ${feature === f.id ? "active" : ""}`}
                  onClick={() => setFeature(f.id)}
                >
                  {f.label}
                </button>
              ))}
            </div>
          )}

          <div className="fc-section-title" style={{ marginTop: 14 }}>Tools</div>
          <div className="fc-game-tools">
            <label className="fc-tool-row">
              <span>Brush</span>
              <input
                type="range" min={1} max={8} value={brushSize}
                onChange={e => setBrushSize(Number(e.target.value))}
              />
              <span>{brushSize}px</span>
            </label>
            <button
              className={`fc-btn ${isEraser ? "primary" : ""}`}
              style={{ marginTop: 6 }}
              onClick={() => setIsEraser(p => !p)}
            >
              {isEraser ? "[ ERASER ]" : "[ BRUSH ]"}
            </button>
          </div>

          <div className="fc-section-title" style={{ marginTop: 14 }}>Your Rank</div>
          <div className="fc-rank-badge" style={{ color: lv.color }}>{lv.label}</div>
          <div className="fc-xp-label">{stats.xp} XP &nbsp;·&nbsp; {stats.gamesPlayed} games</div>
          <div className="fc-xp-track">
            <div className="fc-xp-fill" style={{ width: `${prog.pct}%`, background: lv.color }} />
          </div>
          <div className="fc-xp-sub">{prog.pct}% to next rank</div>
        </div>

        {/* ── Center: canvas ────────────────────────────────────── */}
        <div className="fc-game-canvas-area">
          {phase === "showing" && (
            <div className="fc-ghost-overlay">MEMORIZE &nbsp;{ghostCountdown}s</div>
          )}

          <div className="fc-canvas-stack" style={{ width: W, height: H }}>
            <canvas ref={tplRef} width={W} height={H} className="fc-canvas-tpl" />
            <canvas
              ref={drawRef}
              width={W}
              height={H}
              className="fc-canvas-draw"
              style={{ cursor: phase === "drawing" ? (isEraser ? "cell" : "crosshair") : "default" }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerLeave={onPointerUp}
            />
          </div>

          <div className="fc-game-actions">
            {phase === "idle" && (
              <button className="fc-btn primary" onClick={startGame}>Start Game</button>
            )}
            {phase === "drawing" && (
              <>
                <button
                  className="fc-btn primary"
                  onClick={submitSketch}
                  disabled={predictMutation.isPending}
                >
                  {predictMutation.isPending ? "Analyzing..." : "Submit Sketch"}
                </button>
                <button className="fc-btn" onClick={clearDraw}>Clear</button>
                <button className="fc-btn" onClick={resetGame}>Cancel</button>
              </>
            )}
            {phase === "result" && (
              <button className="fc-btn primary" onClick={resetGame}>Play Again</button>
            )}
          </div>
        </div>

        {/* ── Right: score & leaderboard ────────────────────────── */}
        <div className="fc-game-stats">
          {scoreResult ? (
            <div className="fc-score-section">
              <div className="fc-section-title">Score</div>
              <div className="fc-score-ring">
                <span>{scoreResult.overall}</span>
                <small>/ 100</small>
              </div>
              <div className="fc-score-label">{scoreResult.label}</div>
              {xpGained > 0 && (
                <div className="fc-xp-gained">+{xpGained} XP</div>
              )}
              <div className="fc-score-breakdown">
                <div className="fc-breakdown-row">
                  <span>Coverage</span><span>{scoreResult.recall}%</span>
                </div>
                <div className="fc-breakdown-row">
                  <span>Precision</span><span>{scoreResult.precision}%</span>
                </div>
                <div className="fc-breakdown-row">
                  <span>Best Ever</span><span>{stats.bestScore}%</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="fc-empty" style={{ minHeight: 120, fontSize: "0.7rem", marginBottom: 12 }}>
              Complete a game to see scores
            </div>
          )}

          {/* AI match results for Boost mode */}
          {matchResult?.matches?.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <div className="fc-section-title">AI Match Result</div>
              {(matchResult.matches as any[]).slice(0, 2).map((m, i) => (
                <div key={i} className="fc-mini-match">
                  {m.image_url
                    ? <img src={m.image_url} alt={m.name ?? "match"} />
                    : <div className="fc-mini-ph" />}
                  <div className="fc-mini-info">
                    <strong>{m.name ?? "Unknown"}</strong>
                    <span>{((m.confidence ?? 0) * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Leaderboard */}
          {leaderboard.length > 0 && (
            <div style={{ marginTop: 14 }}>
              <div className="fc-section-title">Best Scores</div>
              {leaderboard.map((entry, i) => (
                <div key={i} className="fc-lb-row">
                  <span className="fc-lb-rank">#{i + 1}</span>
                  <span className="fc-lb-score">{entry.score}%</span>
                  <span className="fc-lb-mode">{entry.mode}</span>
                  <span className="fc-lb-date">{entry.date}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
