/**
 * FACE QUIZ
 * ─────────
 * ZERO drawing. Pure forensic observation + recognition.
 *
 * Each round: a TARGET face is shown on the left.
 * Six candidate faces appear in a grid on the right.
 * The player clicks the one that matches the target.
 * 15-second countdown per round. 5 rounds total.
 *
 * Mechanic: teaches players to READ face structure — the prerequisite
 * skill for drawing accurate sketches.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { FACE_PROFILES, FaceProfile, profileToDataURL } from "./faceUtils";

const ROUNDS = 5;
const TIME_PER_ROUND = 15;

interface Round {
  targetIdx: number;
  options: number[]; // shuffled indices into FACE_PROFILES
}

function buildRounds(): Round[] {
  const rounds: Round[] = [];
  const targets = shuffle([0, 1, 2, 3, 4]).slice(0, ROUNDS); // 5 distinct targets
  for (const t of targets) {
    const others = FACE_PROFILES.map((_, i) => i).filter(i => i !== t);
    const options = shuffle([t, ...shuffle(others).slice(0, 5)]);
    rounds.push({ targetIdx: t, options });
  }
  return rounds;
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

interface Props {
  onGameComplete: (score: number) => void;
}

export default function FaceQuiz({ onGameComplete }: Props) {
  // Pre-render all face images once
  const [faceImgs, setFaceImgs] = useState<string[]>([]);
  useEffect(() => {
    setFaceImgs(FACE_PROFILES.map(p => profileToDataURL(p, 150, "#6a9abf")));
  }, []);

  const [phase, setPhase]         = useState<"intro" | "playing" | "summary">("intro");
  const [rounds, setRounds]       = useState<Round[]>([]);
  const [roundIdx, setRoundIdx]   = useState(0);
  const [timeLeft, setTimeLeft]   = useState(TIME_PER_ROUND);
  const [selected, setSelected]   = useState<number | null>(null); // profile idx chosen
  const [revealed, setRevealed]   = useState(false);
  const [results, setResults]     = useState<boolean[]>([]); // correct/wrong per round
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimer = () => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } };
  useEffect(() => () => clearTimer(), []);

  const startGame = () => {
    const r = buildRounds();
    setRounds(r);
    setRoundIdx(0);
    setResults([]);
    setSelected(null);
    setRevealed(false);
    setTimeLeft(TIME_PER_ROUND);
    setPhase("playing");
  };

  // countdown
  useEffect(() => {
    if (phase !== "playing" || revealed) return;
    clearTimer();
    timerRef.current = setInterval(() => {
      setTimeLeft(t => {
        if (t <= 1) {
          clearTimer();
          setRevealed(true); // timeout = wrong
          setResults(prev => [...prev, false]);
          return 0;
        }
        return t - 1;
      });
    }, 1000);
    return clearTimer;
  }, [phase, roundIdx, revealed]);

  const pick = useCallback((profileIdx: number) => {
    if (revealed) return;
    clearTimer();
    setSelected(profileIdx);
    setRevealed(true);
    const correct = profileIdx === rounds[roundIdx]?.targetIdx;
    setResults(prev => [...prev, correct]);
  }, [revealed, rounds, roundIdx]);

  // Auto-advance after revealing
  useEffect(() => {
    if (!revealed) return;
    const id = setTimeout(() => {
      if (roundIdx >= ROUNDS - 1) {
        setPhase("summary");
      } else {
        setRoundIdx(i => i + 1);
        setSelected(null);
        setRevealed(false);
        setTimeLeft(TIME_PER_ROUND);
      }
    }, 1200);
    return () => clearTimeout(id);
  }, [revealed, roundIdx]);

  // Score when game ends
  useEffect(() => {
    if (phase === "summary") {
      const correct = results.filter(Boolean).length;
      onGameComplete(Math.round((correct / ROUNDS) * 100));
    }
  }, [phase]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Intro ─────────────────────────────────────────────────────────────────
  if (phase === "intro") return (
    <div className="sg-intro">
      <div className="sg-intro-icon">🔍</div>
      <h2>Face Recognition Quiz</h2>
      <p>Study the TARGET face on the left.<br />
         Find its match among 6 candidates on the right.<br />
         <strong>No drawing required.</strong> You have {TIME_PER_ROUND}s per round.</p>
      <p className="sg-intro-note">Trains: forensic observation · face structure recognition</p>
      {faceImgs.length > 0
        ? <button className="fc-btn primary" onClick={startGame}>Start Quiz</button>
        : <span style={{ color: "#6a9abf", fontSize: "0.75rem" }}>Loading faces…</span>}
    </div>
  );

  // ── Summary ───────────────────────────────────────────────────────────────
  if (phase === "summary") {
    const correct = results.filter(Boolean).length;
    const pct = Math.round((correct / ROUNDS) * 100);
    return (
      <div className="sg-intro">
        <div className="sg-intro-icon">📋</div>
        <h2>Quiz Complete</h2>
        <div className="sg-quiz-results">
          {results.map((ok, i) => (
            <div key={i} className={`sg-quiz-result-row ${ok ? "correct" : "wrong"}`}>
              <span>Round {i + 1}</span>
              <span>{ok ? "✓ Correct" : "✗ Missed"}</span>
              <span style={{ color: "#6a9abf", fontSize: "0.7rem" }}>{FACE_PROFILES[rounds[i]?.targetIdx]?.label}</span>
            </div>
          ))}
        </div>
        <div className="sg-final-score">{correct}<small>/{ROUNDS}</small></div>
        <button className="fc-btn primary" onClick={startGame}>Play Again</button>
      </div>
    );
  }

  // ── Playing ───────────────────────────────────────────────────────────────
  const round = rounds[roundIdx];
  if (!round || faceImgs.length === 0) return null;
  const target = FACE_PROFILES[round.targetIdx];

  const timerPct = (timeLeft / TIME_PER_ROUND) * 100;
  const timerColor = timeLeft <= 5 ? "#ff4444" : timeLeft <= 8 ? "#ffd700" : "#00d4ff";

  return (
    <div className="sg-quiz-shell">
      {/* Header */}
      <div className="sg-quiz-header">
        <span className="sg-round-badge">Round {roundIdx + 1} / {ROUNDS}</span>
        <div className="sg-timer-bar">
          <div className="sg-timer-fill" style={{ width: `${timerPct}%`, background: timerColor }} />
        </div>
        <span className="sg-timer-num" style={{ color: timerColor }}>{timeLeft}s</span>
        <span className="sg-score-badge">{results.filter(Boolean).length} / {roundIdx} ✓</span>
      </div>

      <div className="sg-quiz-body">
        {/* Target face */}
        <div className="sg-quiz-target">
          <div className="sg-quiz-label">TARGET FACE</div>
          <div className="sg-target-frame">
            <img src={profileToDataURL(target, 200, "#00d4ff")} alt="target" />
            <div className="sg-target-name">{target.label}</div>
          </div>
          <div className="sg-quiz-instruction">
            Click the face that matches this profile
          </div>
        </div>

        {/* Candidates grid */}
        <div className="sg-quiz-grid">
          {round.options.map(profileIdx => {
            const p = FACE_PROFILES[profileIdx];
            const isCorrect = profileIdx === round.targetIdx;
            const isChosen  = selected === profileIdx;
            let borderColor = "#1a2d48";
            if (revealed) {
              borderColor = isCorrect ? "#00ffcc" : isChosen ? "#ff4444" : "#1a2d48";
            }
            return (
              <button
                key={profileIdx}
                className={`sg-quiz-option ${revealed ? (isCorrect ? "correct" : isChosen ? "wrong" : "") : ""}`}
                style={{ borderColor }}
                onClick={() => pick(profileIdx)}
                disabled={revealed}
              >
                <img src={faceImgs[profileIdx]} alt={p.label} />
                {revealed && isCorrect && <div className="sg-opt-badge correct">✓</div>}
                {revealed && isChosen && !isCorrect && <div className="sg-opt-badge wrong">✗</div>}
              </button>
            );
          })}
        </div>
      </div>

      {revealed && (
        <div className={`sg-reveal-banner ${results[results.length - 1] ? "correct" : "wrong"}`}>
          {results[results.length - 1]
            ? "✓ Correct — moving to next round"
            : timeLeft === 0
            ? "⏱ Time's up! The answer was highlighted."
            : `✗ Incorrect — the correct face was "${target.label}"`}
        </div>
      )}
    </div>
  );
}
