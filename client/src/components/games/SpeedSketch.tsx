/**
 * MEMORY STUDY
 * ────────────
 * Timed pressure game. 3 rounds.
 *
 * Each round:
 *   1. STUDY phase (5s) — a randomised face shown at full opacity
 *   2. DRAW phase  (30s) — face disappears, countdown ticks
 *   3. SCORE phase — accuracy × speed multiplier
 *
 * Fix: canvases are always mounted so refs are never null.
 * Intro / summary are rendered as overlay panels on top of the canvas area.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  FACE_PROFILES,
  FaceProfile,
  drawProfileFace,
  getCanvasPos,
  makeReferenceCanvas,
  pixelScore,
} from "./faceUtils";

const W = 380;
const H = 380;
const STUDY_SECS   = 5;
const DRAW_SECS    = 30;
const TOTAL_ROUNDS = 3;

function pickRoundProfiles(): FaceProfile[] {
  return [...FACE_PROFILES].sort(() => Math.random() - 0.5).slice(0, TOTAL_ROUNDS);
}

function speedMultiplier(timeUsed: number): number {
  const r = timeUsed / DRAW_SECS;
  if (r <= 0.40) return 2.0;
  if (r <= 0.60) return 1.5;
  if (r <= 0.80) return 1.25;
  return 1.0;
}

interface Props { onGameComplete: (score: number) => void; }

export default function SpeedSketch({ onGameComplete }: Props) {
  const tplRef    = useRef<HTMLCanvasElement>(null);
  const drawRef   = useRef<HTMLCanvasElement>(null);
  const isDrawRef = useRef(false);
  const lastRef   = useRef<{ x: number; y: number } | null>(null);
  const timerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  // Store live values for timer callbacks (avoid stale closures)
  const profilesRef = useRef<FaceProfile[]>([]);
  const roundRef    = useRef(0);

  const [phase, setPhase]  = useState<"intro" | "study" | "draw" | "scored" | "summary">("intro");
  const [round, setRound]  = useState(0);
  const [profiles, setProfiles] = useState<FaceProfile[]>([]);
  const [studyLeft, setStudyLeft] = useState(STUDY_SECS);
  const [drawLeft,  setDrawLeft]  = useState(DRAW_SECS);
  const [drawStart, setDrawStart] = useState(0);
  const [roundResults, setRoundResults] = useState<{ raw: number; mult: number; final: number }[]>([]);
  const [brushSize, setBrushSize] = useState(2);
  const [lastRound, setLastRound] = useState<{ raw: number; mult: number; final: number } | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
  }, []);
  useEffect(() => () => clearTimer(), [clearTimer]);

  // ── Draw template via effect (runs AFTER canvas mounts in DOM) ────────────
  useEffect(() => {
    if (phase !== "study") return;
    const tpl = tplRef.current;
    if (!tpl) return;
    const profs = profilesRef.current;
    const r = roundRef.current;
    if (!profs[r]) return;
    const ctx = tpl.getContext("2d")!;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0a0f1a";
    ctx.fillRect(0, 0, W, H);
    drawProfileFace(ctx, W / 2, H / 2, profs[r], 1.0, "#00d4ff");
  }, [phase, round]); // re-runs each time phase flips to "study" for a new round

  // ── Clear draw canvas ─────────────────────────────────────────────────────
  const clearDraw = useCallback(() => {
    const dc = drawRef.current; if (!dc) return;
    dc.getContext("2d")!.clearRect(0, 0, W, H);
  }, []);

  const hideTemplate = useCallback(() => {
    const tpl = tplRef.current; if (!tpl) return;
    const ctx = tpl.getContext("2d")!;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0a0f1a";
    ctx.fillRect(0, 0, W, H);
  }, []);

  // ── Score and advance ─────────────────────────────────────────────────────
  const scoreRound = useCallback((roundIdx: number, timeUsed: number) => {
    clearTimer();
    const dc = drawRef.current; if (!dc) return;
    const profs = profilesRef.current;
    const ref = makeReferenceCanvas(profs[roundIdx], W, H);
    const { overall } = pixelScore(dc, ref);
    const mult  = speedMultiplier(timeUsed);
    const final = Math.min(100, Math.round(overall * mult));
    const res   = { raw: overall, mult, final };
    setLastRound(res);
    setRoundResults(prev => {
      const updated = [...prev, res];
      if (roundIdx >= TOTAL_ROUNDS - 1) {
        const avg = Math.min(100, Math.round(updated.reduce((s, r) => s + r.final, 0) / updated.length));
        setTimeout(() => { setPhase("summary"); onGameComplete(avg); }, 1200);
      } else {
        const next = roundIdx + 1;
        roundRef.current = next;
        setRound(next);
        clearDraw();
        setTimeout(() => beginStudy(), 1400);
      }
      return updated;
    });
    setPhase("scored");
  }, [clearTimer, clearDraw, onGameComplete]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Study countdown (runs entirely via interval) ──────────────────────────
  const beginStudy = useCallback(() => {
    clearDraw();
    setLastRound(null);
    setStudyLeft(STUDY_SECS);
    setPhase("study"); // useEffect above draws the template after this renders

    let remaining = STUDY_SECS;
    clearTimer();
    timerRef.current = setInterval(() => {
      remaining--;
      setStudyLeft(remaining);
      if (remaining <= 0) {
        clearTimer();
        hideTemplate();
        setDrawLeft(DRAW_SECS);
        const start = Date.now();
        setDrawStart(start);
        setPhase("draw");

        let dl = DRAW_SECS;
        timerRef.current = setInterval(() => {
          dl--;
          setDrawLeft(dl);
          if (dl <= 0) {
            const used = Math.round((Date.now() - start) / 1000);
            scoreRound(roundRef.current, used);
          }
        }, 1000);
      }
    }, 1000);
  }, [clearTimer, clearDraw, hideTemplate, scoreRound]);

  // ── Drawing events ────────────────────────────────────────────────────────
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
    const last = lastRef.current;
    if (!last) { lastRef.current = pos; return; }
    ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(pos.x, pos.y);
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = brushSize;
    ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.stroke();
    lastRef.current = pos;
  }, [phase, brushSize]);

  const onUp = useCallback(() => { isDrawRef.current = false; lastRef.current = null; }, []);

  const submitNow = useCallback(() => {
    const used = Math.round((Date.now() - drawStart) / 1000);
    scoreRound(roundRef.current, used);
  }, [drawStart, scoreRound]);

  const startGame = () => {
    const profs = pickRoundProfiles();
    profilesRef.current = profs;
    roundRef.current = 0;
    setProfiles(profs);
    setRound(0);
    setRoundResults([]);
    setLastRound(null);
    clearDraw();
    beginStudy();
  };

  const drawPct   = (drawLeft  / DRAW_SECS)   * 100;
  const studyPct  = (studyLeft / STUDY_SECS)  * 100;
  const timerColor = drawLeft <= 8 ? "#ff4444" : drawLeft <= 15 ? "#ffd700" : "#00d4ff";

  // ── Canvas area — always rendered so refs are always valid ────────────────
  const canvasBlock = (
    <div className="sg-canvas-col">
      <div className="sg-canvas-stack" style={{ width: W, height: H }}>
        <canvas ref={tplRef}  width={W} height={H} className="sg-canvas-bg" />
        <canvas ref={drawRef} width={W} height={H} className="sg-canvas-fg"
          style={{ cursor: phase === "draw" ? "crosshair" : "default" }}
          onPointerDown={onDown} onPointerMove={onMove}
          onPointerUp={onUp}    onPointerLeave={onUp} />
      </div>
      {phase === "draw" && (
        <div className="sg-canvas-actions">
          <button className="fc-btn primary" onClick={submitNow}>Submit Now</button>
          <button className="fc-btn" onClick={clearDraw}>Clear</button>
          <div className="sg-tool-row">
            <span>Brush</span>
            <input type="range" min={1} max={6} value={brushSize}
              onChange={e => setBrushSize(+e.target.value)} />
            <span>{brushSize}px</span>
          </div>
        </div>
      )}
    </div>
  );

  // ── Intro ─────────────────────────────────────────────────────────────────
  if (phase === "intro") return (
    <div className="sg-speed-shell">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, alignItems: "start" }}>
        {canvasBlock}
        <div className="sg-intro" style={{ margin: 0 }}>
          <div className="sg-intro-icon">🧠</div>
          <h2>Memory Study</h2>
          <p>A face appears for <strong>{STUDY_SECS} seconds</strong>. Study it.<br />
             Then it vanishes — draw it in <strong>{DRAW_SECS} seconds</strong>.<br />
             Submit early for a <strong>speed bonus (up to ×2.0)</strong>.<br />
             3 different faces. No second chances.</p>
          <p className="sg-intro-note">Trains: memory · speed · reconstruction</p>
          <button className="fc-btn primary" onClick={startGame}>Start Race</button>
        </div>
      </div>
    </div>
  );

  // ── Summary ───────────────────────────────────────────────────────────────
  if (phase === "summary") {
    const avg = Math.min(100, Math.round(roundResults.reduce((s, r) => s + r.final, 0) / roundResults.length));
    return (
      <div className="sg-speed-shell">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, alignItems: "start" }}>
          {canvasBlock}
          <div className="sg-intro" style={{ margin: 0 }}>
            <div className="sg-intro-icon">🏆</div>
            <h2>Race Complete</h2>
            <div className="sg-speed-results">
              {roundResults.map((r, i) => (
                <div key={i} className="sg-speed-row">
                  <span>Round {i + 1}</span>
                  <span>Accuracy: {r.raw}%</span>
                  <span className="sg-mult">×{r.mult.toFixed(2)}</span>
                  <span className="sg-final-pts">{r.final}pts</span>
                </div>
              ))}
            </div>
            <div className="sg-final-score">{avg}<small>/100</small></div>
            <button className="fc-btn primary" onClick={startGame}>Race Again</button>
          </div>
        </div>
      </div>
    );
  }

  // ── Playing ───────────────────────────────────────────────────────────────
  return (
    <div className="sg-speed-shell">
      <div className="sg-speed-header">
        <span className="sg-round-badge">Round {round + 1} / {TOTAL_ROUNDS}
          {profiles[round] && <span style={{ color: "#4e7697", marginLeft: 8 }}>— {profiles[round].label}</span>}
        </span>

        {phase === "study" && (
          <div className="sg-study-banner">
            STUDY THE FACE — {studyLeft}s
            <div className="sg-study-bar">
              <div style={{ width: `${studyPct}%`, background: "#00d4ff", height: "100%", borderRadius: 2, transition: "width 1s linear" }} />
            </div>
          </div>
        )}
        {phase === "draw" && (
          <div className="sg-draw-banner" style={{ color: timerColor }}>
            DRAW FROM MEMORY — {drawLeft}s
            <div className="sg-study-bar">
              <div style={{ width: `${drawPct}%`, background: timerColor, height: "100%", borderRadius: 2, transition: "width 1s linear" }} />
            </div>
          </div>
        )}
        {phase === "scored" && lastRound && (
          <div className="sg-scored-banner">
            {lastRound.raw}% accuracy × {lastRound.mult.toFixed(2)} speed = <strong>{lastRound.final} pts</strong>
            {round < TOTAL_ROUNDS - 1 ? " — next round loading…" : ""}
          </div>
        )}
      </div>

      <div className="sg-speed-body">
        {canvasBlock}

        <div className="sg-speed-scores">
          <div className="fc-section-title">Rounds</div>
          {Array.from({ length: TOTAL_ROUNDS }).map((_, i) => {
            const r = roundResults[i];
            return (
              <div key={i} className={`sg-speed-score-row ${i === round && phase !== "summary" ? "current" : ""}`}>
                <span>Round {i + 1}</span>
                {r ? <span style={{ color: "#00ffcc" }}>{r.final} pts</span>
                   : i === round ? <span style={{ color: "#ffd700" }}>← now</span>
                   : <span style={{ color: "#4e7697" }}>—</span>}
              </div>
            );
          })}

          <div className="sg-speed-mult-guide">
            <div className="fc-section-title" style={{ marginTop: 14 }}>Speed Bonus</div>
            {[["< 12s", "×2.0", "#00ffcc"], ["< 18s", "×1.5", "#00d4ff"], ["< 24s", "×1.25", "#ffd700"], ["30s", "×1.0", "#6a9abf"]].map(([t, m, c]) => (
              <div key={t} className="sg-mult-row"><span>{t}</span><span style={{ color: c }}>{m}</span></div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
