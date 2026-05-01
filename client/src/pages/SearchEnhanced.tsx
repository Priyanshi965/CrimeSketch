import React, { useRef, useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { AlertCircle, CheckCircle, AlertTriangle, Zap } from 'lucide-react';
import { useRealtimeSketchFeedback } from '@/hooks/useRealtimeSketchFeedback';
import { motion, AnimatePresence } from 'framer-motion';

interface FeedbackItem {
  type: 'issue' | 'suggestion';
  text: string;
}

export default function SearchEnhanced() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [showExplainability, setShowExplainability] = useState(false);
  const [feedbackItems, setFeedbackItems] = useState<FeedbackItem[]>([]);
  const [animateResults, setAnimateResults] = useState(false);

  const {
    feedback,
    matches,
    edgeMap,
    preprocessedImage,
    inferenceTime,
    isLoading,
    error,
    processSketchFrame,
    reset,
  } = useRealtimeSketchFeedback({
    enabled: isDrawing,
    debounceMs: 300,
  });

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = 512;
    canvas.height = 512;

    // Fill with white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Set drawing style
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }, []);

  // Handle canvas drawing
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();

    // Send frame for real-time feedback every 300ms
    canvas.toBlob((blob) => {
      if (blob) {
        processSketchFrame(blob);
      }
    }, 'image/png');
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  // Update feedback items when feedback changes
  useEffect(() => {
    if (feedback) {
      const items: FeedbackItem[] = [];

      feedback.issues.forEach((issue) => {
        items.push({ type: 'issue', text: issue });
      });

      feedback.suggestions.forEach((suggestion) => {
        items.push({ type: 'suggestion', text: suggestion });
      });

      setFeedbackItems(items);
    }
  }, [feedback]);

  // Animate results when matches appear
  useEffect(() => {
    if (matches.length > 0) {
      setAnimateResults(true);
    }
  }, [matches]);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    reset();
    setFeedbackItems([]);
    setAnimateResults(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Sketch Analysis</h1>
          <p className="text-slate-400">Real-time feedback as you draw</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Canvas Section */}
          <div className="lg:col-span-2">
            <Card className="bg-slate-800 border-slate-700 p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Draw Facial Sketch</h2>

              {/* Canvas */}
              <div className="mb-6 border-2 border-dashed border-slate-600 rounded-lg overflow-hidden bg-white">
                <canvas
                  ref={canvasRef}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseUp}
                  className="w-full cursor-crosshair"
                />
              </div>

              {/* Canvas Controls */}
              <div className="flex gap-3 mb-6">
                <Button
                  onClick={clearCanvas}
                  variant="outline"
                  className="flex-1 border-slate-600 text-slate-200 hover:bg-slate-700"
                >
                  Clear Canvas
                </Button>
                <Button className="flex-1 bg-amber-500 hover:bg-amber-600 text-white">
                  Search
                </Button>
              </div>

              {/* Explainability Toggle */}
              <div className="flex items-center justify-between p-3 bg-slate-700 rounded-lg">
                <span className="text-sm text-slate-300">Show Explainability</span>
                <button
                  onClick={() => setShowExplainability(!showExplainability)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    showExplainability ? 'bg-amber-500' : 'bg-slate-600'
                  }`}
                />
              </div>
            </Card>

            {/* Explainability Section */}
            <AnimatePresence>
              {showExplainability && (edgeMap || preprocessedImage) && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="mt-6 grid grid-cols-2 gap-4"
                >
                  {edgeMap && (
                    <Card className="bg-slate-800 border-slate-700 p-4">
                      <h3 className="text-sm font-semibold text-white mb-3">Edge Map</h3>
                      <img
                        src={edgeMap}
                        alt="Edge Map"
                        className="w-full rounded border border-slate-600"
                      />
                      <p className="text-xs text-slate-400 mt-2">
                        Shows detected edges and features
                      </p>
                    </Card>
                  )}

                  {preprocessedImage && (
                    <Card className="bg-slate-800 border-slate-700 p-4">
                      <h3 className="text-sm font-semibold text-white mb-3">Preprocessed</h3>
                      <img
                        src={preprocessedImage}
                        alt="Preprocessed"
                        className="w-full rounded border border-slate-600"
                      />
                      <p className="text-xs text-slate-400 mt-2">
                        Normalized and enhanced
                      </p>
                    </Card>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Feedback & Results Section */}
          <div className="space-y-6">
            {/* Real-Time Feedback */}
            {feedback && (
              <Card className="bg-slate-800 border-slate-700 p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Sketch Quality</h3>

                {/* Quality Scores */}
                <div className="space-y-4 mb-6">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-slate-300">Overall Quality</span>
                      <span className="text-sm font-semibold text-amber-400">
                        {(feedback.overall_quality * 100).toFixed(0)}%
                      </span>
                    </div>
                    <Progress
                      value={feedback.overall_quality * 100}
                      className="h-2 bg-slate-700"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-slate-300">Symmetry</span>
                      <span className="text-sm text-slate-400">
                        {(feedback.symmetry_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <Progress
                      value={feedback.symmetry_score * 100}
                      className="h-2 bg-slate-700"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-slate-300">Completeness</span>
                      <span className="text-sm text-slate-400">
                        {(feedback.completeness_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <Progress
                      value={feedback.completeness_score * 100}
                      className="h-2 bg-slate-700"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-slate-300">Clarity</span>
                      <span className="text-sm text-slate-400">
                        {(feedback.clarity_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <Progress
                      value={feedback.clarity_score * 100}
                      className="h-2 bg-slate-700"
                    />
                  </div>
                </div>

                {/* Issues & Suggestions */}
                <div className="space-y-2">
                  <AnimatePresence>
                    {feedbackItems.map((item, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 10 }}
                        className={`flex items-start gap-2 p-2 rounded text-sm ${
                          item.type === 'issue'
                            ? 'bg-red-900/20 text-red-300'
                            : 'bg-blue-900/20 text-blue-300'
                        }`}
                      >
                        {item.type === 'issue' ? (
                          <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        ) : (
                          <Zap className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        )}
                        <span>{item.text}</span>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>

                {/* Inference Time */}
                {inferenceTime > 0 && (
                  <div className="mt-4 pt-4 border-t border-slate-700 text-xs text-slate-400">
                    <span>⚡ Inference: {inferenceTime.toFixed(1)}ms</span>
                  </div>
                )}
              </Card>
            )}

            {/* Top Matches */}
            {matches.length > 0 && (
              <Card className="bg-slate-800 border-slate-700 p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Top Matches</h3>

                <AnimatePresence>
                  {matches.map((match, idx) => (
                    <motion.div
                      key={match.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.1 }}
                      className="mb-3 p-3 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors cursor-pointer"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <p className="font-semibold text-white text-sm">{match.name}</p>
                          <p className="text-xs text-slate-400">ID: {match.id}</p>
                        </div>
                        <div className="text-right">
                          <p className="font-bold text-amber-400">
                            {(match.confidence * 100).toFixed(1)}%
                          </p>
                          <CheckCircle className="w-4 h-4 text-green-400 mt-1" />
                        </div>
                      </div>

                      <Progress
                        value={match.confidence * 100}
                        className="h-1.5 bg-slate-600"
                      />
                    </motion.div>
                  ))}
                </AnimatePresence>
              </Card>
            )}

            {/* Loading State */}
            {isLoading && (
              <Card className="bg-slate-800 border-slate-700 p-6">
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-amber-500" />
                </div>
                <p className="text-center text-slate-400 text-sm mt-3">
                  Analyzing sketch...
                </p>
              </Card>
            )}

            {/* Error State */}
            {error && (
              <Card className="bg-red-900/20 border-red-700 p-6">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-semibold text-red-300">Error</p>
                    <p className="text-sm text-red-200">{error}</p>
                  </div>
                </div>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
