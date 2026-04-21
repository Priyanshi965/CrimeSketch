/**
 * SketchGame — container for 4 independent forensic drawing games.
 *
 * Each game has a completely different mechanic:
 *   1. Stroke Trainer  — sequential segment-by-segment drawing
 *   2. Face Quiz       — click-to-match recognition (no drawing)
 *   3. Speed Sketch    — timed memory race (3 rounds, speed bonus)
 *   4. Match Boost     — free draw → real AI pipeline → match results
 *
 * This component owns: XP/level, leaderboard, mode switching.
 * Games report back via onGameComplete(score).
 */

import { useState } from "react";
import StrokeTrainer from "./games/StrokeTrainer";
import FaceQuiz      from "./games/FaceQuiz";
import SpeedSketch   from "./games/SpeedSketch";
import MatchBoost    from "./games/MatchBoost";

// ─── Persistence ──────────────────────────────────────────────────────────────

type GameMode = "stroke" | "quiz" | "speed" | "boost";

interface GameStats {
  xp: number;
  gamesPlayed: number;
  bestScore: number;
  history: { score: number; mode: GameMode; date: string; xpGained: number }[];
}

const STATS_KEY = "crimesketch_game_stats";

function loadStats(): GameStats {
  try { const r = localStorage.getItem(STATS_KEY); if (r) return JSON.parse(r); } catch {}
  return { xp: 0, gamesPlayed: 0, bestScore: 0, history: [] };
}
function saveStats(s: GameStats) { localStorage.setItem(STATS_KEY, JSON.stringify(s)); }

const LEVELS = [
  { minXp: 0,    label: "Beginner Artist",  color: "#6a9abf" },
  { minXp: 200,  label: "Sketch Analyst",   color: "#00d4ff" },
  { minXp: 500,  label: "Forensic Expert",  color: "#00ffcc" },
  { minXp: 1000, label: "CID Officer",      color: "#ffd700" },
];

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
  return { pct: Math.min(100, Math.round(((xp - cur) / (nxt - cur)) * 100)), cur, nxt };
}

// ─── Game metadata ────────────────────────────────────────────────────────────

const GAMES: { id: GameMode; label: string; subtitle: string; icon: string }[] = [
  { id: "speed",  label: "Memory Study",    subtitle: "Study and draw from memory",    icon: "🧠" },
  { id: "stroke", label: "Stroke Trainer",  subtitle: "Step-by-step face segments",   icon: "🖊" },
  { id: "quiz",   label: "Face Quiz",       subtitle: "Click the matching face",       icon: "🔍" },
  { id: "boost",  label: "Camera Match",     subtitle: "Match sketch or camera photo", icon: "🎯" },
];

// ─── Component ────────────────────────────────────────────────────────────────

export default function SketchGame() {
  const [mode, setMode]   = useState<GameMode>("speed");
  const [stats, setStats] = useState<GameStats>(loadStats);
  const [lastResult, setLastResult] = useState<{ score: number; xp: number } | null>(null);
  // key forces game remount on mode switch
  const [gameKey, setGameKey] = useState(0);

  const lv   = getLevel(stats.xp);
  const prog = xpProgress(stats.xp);
  const leaderboard = [...stats.history].sort((a, b) => b.score - a.score).slice(0, 5);

  const handleModeSwitch = (m: GameMode) => {
    setMode(m);
    setLastResult(null);
    setGameKey(k => k + 1);
  };

  const handleGameComplete = (score: number) => {
    const gained = Math.max(5, Math.round(score * 1.8));
    const newStats: GameStats = {
      xp: stats.xp + gained,
      gamesPlayed: stats.gamesPlayed + 1,
      bestScore: Math.max(stats.bestScore, score),
      history: [
        { score, mode, date: new Date().toLocaleDateString(), xpGained: gained },
        ...stats.history.slice(0, 14),
      ],
    };
    setStats(newStats);
    saveStats(newStats);
    setLastResult({ score, xp: gained });
  };

  return (
    <div className="sg-shell">
      {/* ── Mode selector ─────────────────────────────────────────────── */}
      <div className="sg-mode-bar">
        {GAMES.map(g => (
          <button
            key={g.id}
            className={`sg-mode-btn ${mode === g.id ? "active" : ""}`}
            onClick={() => handleModeSwitch(g.id)}
          >
            <span className="sg-mode-icon">{g.icon}</span>
            <span className="sg-mode-label">{g.label}</span>
            <span className="sg-mode-sub">{g.subtitle}</span>
          </button>
        ))}
      </div>

      <div className="sg-main">
        {/* ── Game area ───────────────────────────────────────────────── */}
        <div className="sg-game-area">
          {mode === "stroke" && <StrokeTrainer key={gameKey} onGameComplete={handleGameComplete} />}
          {mode === "quiz"   && <FaceQuiz      key={gameKey} onGameComplete={handleGameComplete} />}
          {mode === "speed"  && <SpeedSketch   key={gameKey} onGameComplete={handleGameComplete} />}
          {mode === "boost"  && <MatchBoost    key={gameKey} onGameComplete={handleGameComplete} />}
        </div>

        {/* ── Side panel: XP + leaderboard ────────────────────────────── */}
        <div className="sg-side-panel">
          {/* Last result */}
          {lastResult && (
            <div className="sg-last-result">
              <div className="sg-last-score">{lastResult.score}<small>/100</small></div>
              <div className="sg-last-xp">+{lastResult.xp} XP</div>
            </div>
          )}

          {/* Rank */}
          <div className="fc-section-title">Rank</div>
          <div className="sg-rank-name" style={{ color: lv.color }}>{lv.label}</div>
          <div className="sg-xp-row">{stats.xp} XP &nbsp;·&nbsp; {stats.gamesPlayed} games</div>
          <div className="sg-xp-track">
            <div className="sg-xp-fill" style={{ width: `${prog.pct}%`, background: lv.color }} />
          </div>
          <div className="sg-xp-sub">{prog.pct}% → next rank</div>

          {/* Level ladder */}
          <div className="sg-levels">
            {LEVELS.map(l => (
              <div key={l.label} className={`sg-level-row ${stats.xp >= l.minXp ? "earned" : ""}`}
                style={{ color: stats.xp >= l.minXp ? l.color : "#4e7697" }}>
                <span>{stats.xp >= l.minXp ? "●" : "○"}</span>
                <span>{l.label}</span>
                <span>{l.minXp} XP</span>
              </div>
            ))}
          </div>

          {/* Leaderboard */}
          {leaderboard.length > 0 && (
            <>
              <div className="fc-section-title" style={{ marginTop: 14 }}>Best Scores</div>
              {leaderboard.map((e, i) => (
                <div key={i} className="sg-lb-row">
                  <span className="sg-lb-rank">#{i + 1}</span>
                  <span className="sg-lb-score">{e.score}%</span>
                  <span className="sg-lb-mode">
                    {GAMES.find(g => g.id === e.mode)?.icon ?? ""} {e.mode}
                  </span>
                  <span className="sg-lb-date">{e.date}</span>
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
