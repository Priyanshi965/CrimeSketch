/**
 * SPEED SKETCH
 * ────────────
 * Timed pressure game. 3 rounds.
 *
 * Each round:
 *   1. STUDY phase (5s) — a randomised face shown at full opacity
 *   2. DRAW phase  (30s) — face disappears, countdown ticks
 *   3. SCORE phase — accuracy × speed multiplier
 *
 * Faster submissions earn a speed bonus (up to ×2).
 * Final score = average(round_scores × speed_multiplier).
 *
 * Mechanic: urgency + surprise — different from all other games.
 *           No guide visible during drawing. Time is the enemy.
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
const STUDY_SECS  = 5;
const DRAW_SECS   = 30;
const TOTAL_ROUNDS = 3;

function pickRoundProfiles(): FaceProfile[] {
  const shuffled = [...FACE_PROFILES].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, TOTAL_ROUNDS);
}

function speedMultiplier(timeUsed: number): number {
  const ratio = timeUsed / DRAW_SECS;
  if (ratio <= 0.40) return 2.0;
  if (ratio <= 0.60) return 1.5;
  if (ratio <= 0.80) return 1.25;
  return 1.0;
}

interface Props {
  onGameComplete: (score: number) => void;
}

export default function SpeedSketch({ onGameComplete }: Props) {
  const tplRef  = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef<HTMLCanvasElement>(null);
  const isDrawRef = useRef(false);
  const lastRef   = useRef<{ x: number; y: number } | null>(null);
  const timerRef  = useRef<ReturnType<typeof setInterval> | null>(null);

  const [phase, setPhase]  = useState<"intro" | "study" | "draw" | "scored" | "summary">("intro");
  const [round, setRound]  = useState(0);
  const [profiles, setProfiles] = useState<FaceProfile[]>([]);
  const [studyLeft, setStudyLeft] = useState(STUDY_SECS);
  const [drawLeft,  setDrawLeft]  = useState(DRAW_SECS);
  const [drawStart, setDrawStart] = useState(0);
  const [roundResults, setRoundResults] = useState<{ raw: number; mult: number; final: number }[]>([]);
  const [brushSize, setBrushSize] = useState(2);
  const [lastRound, setLastRound] = useState<{ raw: number; mult: number; final: number } | null>(null);

  const clearTimer = () => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } };
  useEffect(() => () => clearTimer(), []);

  const showTemplate = useCallback((profile: FaceProfile, alpha: number) => {
    const tpl = tplRef.current; if (!tpl) return;
    const ctx = tpl.getContext("2d")!;
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#0a0f1a"; ctx.fillRect(0, 0, W, H);
    drawProfileFace(ctx, W / 2, H / 2, profile, alpha, "#00d4ff");
  }, []);

  const clearDraw = useCallback(() => {
    const dc = drawRef.current; if (!dc) return;
    dc.getContext("2d")!.clearRect(0, 0, W, H);
  }, []);

  // ── Start a round ─────────────────────────────────────────────────────────
  const startRound = useCallback((roundIdx: number, profs: FaceProfile[]) => {
    clearDraw();
    setLastRound(null);
    showTemplate(profs[roundIdx], 1.0);
    setStudyLeft(STUDY_SECS);
    setPhase("study");

    let remaining = STUDY_SECS;
    clearTimer();
    timerRef.current = setInterval(() => {
      remaining--;
      setStudyLeft(remaining);
      if (remaining <= 0) {
        clearTimer();
        // Hide template, start drawing phase
        const tpl = tplRef.current;
        if (tpl) { const ctx = tpl.getContext("2d")!; ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#0a0f1a"; ctx.fillRect(0, 0, W, H); }
        setDrawLeft(DRAW_SECS);
        setDrawStart(Date.now());
        setPhase("draw");

        let dl = DRAW_SECS;
        timerRef.current = setInterval(() => {
          dl--;
          setDrawLeft(dl);
          if (dl <= 0) {
            clearTimer();
            autoSubmit(roundIdx, profs);
          }
        }, 1000);
      }
    }, 1000);
  }, [clearDraw, showTemplate]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Auto-submit when time runs out ────────────────────────────────────────
  const autoSubmit = useCallback((roundIdx: number, profs: FaceProfile[]) => {
    clearTimer();
    const dc = drawRef.current; if (!dc) return;
    const ref = makeReferenceCanvas(profs[roundIdx], W, H);
    const { overall } = pixelScore(dc, ref);
    const timeUsed = DRAW_SECS; // full time used
    const mult = speedMultiplier(timeUsed);
    const final = Math.round(overall * mult);
    const res = { raw: overall, mult, final };
    setLastRound(res);
    setRoundResults(prev => {
      const updated = [...prev, res];
      if (roundIdx >= TOTAL_ROUNDS - 1) {
        const avg = Math.round(updated.reduce((s, r) => s + r.final, 0) / updated.length);
        setTimeout(() => { setPhase("summary"); onGameComplete(Math.min(100, avg)); }, 1200);
      } else {
        setTimeout(() => startRound(roundIdx + 1, profs), 1500);
        setRound(roundIdx + 1);
      }
      return updated;
    });
    setPhase("scored");
  }, [onGameComplete, startRound]);

  // ── Manual submit ─────────────────────────────────────────────────────────
  const submit = useCallback(() => {
    clearTimer();
    const timeUsed = Math.round((Date.now() - drawStart) / 1000);
    const dc = drawRef.current; if (!dc) return;
    const ref = makeReferenceCanvas(profiles[round], W, H);
    const { overall } = pixelScore(dc, ref);
    const mult = speedMultiplier(timeUsed);
    const final = Math.round(overall * mult);
    const res = { raw: overall, mult, final };
    setLastRound(res);
    setRoundResults(prev => {
      const updated = [...prev, res];
      if (round >= TOTAL_ROUNDS - 1) {
        const avg = Math.round(updated.reduce((s, r) => s + r.final, 0) / updated.length);
        setTimeout(() => { setPhase("summary"); onGameComplete(Math.min(100, avg)); }, 1400);
      } else {
        setTimeout(() => { startRound(round + 1, profiles); setRound(r => r + 1); }, 1500);
      }
      return updated;
    });
    setPhase("scored");
  }, [drawStart, profiles, round, onGameComplete, startRound]);

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
    const last = lastRef.current; if (!last) { lastRef.current = pos; return; }
    ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(pos.x, pos.y);
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = brushSize;
    ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.stroke();
    lastRef.current = pos;
  }, [phase, brushSize]);

  const onUp = useCallback(() => { isDrawRef.current = false; lastRef.current = null; }, []);

  const startGame = () => {
    const profs = pickRoundProfiles();
    setProfiles(profs);
    setRound(0);
    setRoundResults([]);
    setLastRound(null);
    startRound(0, profs);
  };

  // ── Intro ─────────────────────────────────────────────────────────────────
  if (phase === "intro") return (
    <div className="sg-intro">
      <div className="sg-intro-icon">⚡</div>
      <h2>Speed Sketch</h2>
      <p>A face appears for <strong>{STUDY_SECS} seconds</strong>. Study it.<br />
         Then it vanishes — draw it in <strong>{DRAW_SECS} seconds</strong>.<br />
         Finish early for a <strong>speed bonus (up to ×2)</strong>.<br />
         3 different faces. No second chances.</p>
      <p className="sg-intro-note">Trains: memory · speed · cross-modal reconstruction</p>
      <button className="fc-btn primary" onClick={startGame}>Start Race</button>
    </div>
  );

  // ── Summary ───────────────────────────────────────────────────────────────
  if (phase === "summary") {
    const avg = Math.min(100, Math.round(roundResults.reduce((s, r) => s + r.final, 0) / roundResults.length));
    return (
      <div className="sg-intro">
        <div className="sg-intro-icon">🏆</div>
        <h2>Race Complete</h2>
        <div className="sg-speed-results">
          {roundResults.map((r, i) => (
            <div key={i} className="sg-speed-row">
              <span>Round {i + 1} — {profiles[i]?.label}</span>
              <span>Accuracy: {r.raw}%</span>
              <span className="sg-mult">×{r.mult.toFixed(2)}</span>
              <span className="sg-final-pts">{Math.min(100, r.final)}pts</span>
            </div>
          ))}
        </div>
        <div className="sg-final-score">{avg}<small>/100</small></div>
        <button className="fc-btn primary" onClick={startGame}>Race Again</button>
      </div>
    );
  }

  const drawPct = (drawLeft / DRAW_SECS) * 100;
  const studyPct = (studyLeft / STUDY_SECS) * 100;
  const timerColor = drawLeft <= 8 ? "#ff4444" : drawLeft <= 15 ? "#ffd700" : "#00d4ff";

  return (
    <div className="sg-speed-shell">
      {/* Round indicator */}
      <div className="sg-speed-header">
        <span className="sg-round-badge">Round {round + 1} / {TOTAL_ROUNDS}</span>
        {phase === "study" && (
          <div className="sg-study-banner">
            STUDY THE FACE — {studyLeft}s
            <div className="sg-study-bar"><div style={{ width: `${studyPct}%`, background: "#00d4ff" }} /></div>
          </div>
        )}
        {phase === "draw" && (
          <div className="sg-draw-banner" style={{ color: timerColor }}>
            DRAW! — {drawLeft}s remaining
            <div className="sg-study-bar"><div style={{ width: `${drawPct}%`, background: timerColor }} /></div>
          </div>
        )}
        {phase === "scored" && lastRound && (
          <div className="sg-scored-banner">
            {lastRound.raw}% accuracy × {lastRound.mult.toFixed(2)} speed = <strong>{Math.min(100, lastRound.final)} pts</strong>
          </div>
        )}
      </div>

      <div className="sg-speed-body">
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
              <button className="fc-btn primary" onClick={submit}>Submit Now</button>
              <button className="fc-btn" onClick={clearDraw}>Clear</button>
              <div className="sg-tool-row">
                <span>Brush</span>
                <input type="range" min={1} max={6} value={brushSize} onChange={e => setBrushSize(+e.target.value)} />
                <span>{brushSize}px</span>
              </div>
            </div>
          )}
        </div>

        {/* Round scoreboard */}
        <div className="sg-speed-scores">
          <div className="fc-section-title">Scores</div>
          {Array.from({ length: TOTAL_ROUNDS }).map((_, i) => {
            const r = roundResults[i];
            return (
              <div key={i} className={`sg-speed-score-row ${i === round && phase !== "summary" ? "current" : ""}`}>
                <span>Round {i + 1}</span>
                {r ? (
                  <span style={{ color: "#00ffcc" }}>{Math.min(100, r.final)} pts</span>
                ) : i === round ? (
                  <span style={{ color: "#ffd700" }}>← now</span>
                ) : (
                  <span style={{ color: "#4e7697" }}>—</span>
                )}
              </div>
            );
          })}
          <div className="sg-speed-mult-guide">
            <div className="fc-section-title" style={{ marginTop: 14 }}>Speed Bonus</div>
            <div className="sg-mult-row"><span>&lt; 12s</span><span style={{ color: "#00ffcc" }}>×2.0</span></div>
            <div className="sg-mult-row"><span>&lt; 18s</span><span style={{ color: "#00d4ff" }}>×1.5</span></div>
            <div className="sg-mult-row"><span>&lt; 24s</span><span style={{ color: "#ffd700" }}>×1.25</span></div>
            <div className="sg-mult-row"><span>30s</span><span style={{ color: "#6a9abf" }}>×1.0</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}
