import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, AlertCircle, TrendingUp } from 'lucide-react';

interface Match {
  id: number;
  name: string;
  confidence: number;
  distance: number;
  image_url: string;
}

interface ResultsAnimationProps {
  matches: Match[];
  isLoading: boolean;
  error?: string;
  onMatchSelect?: (match: Match) => void;
}

export const ResultsAnimation: React.FC<ResultsAnimationProps> = ({
  matches,
  isLoading,
  error,
  onMatchSelect,
}) => {
  const [selectedMatch, setSelectedMatch] = useState<Match | null>(null);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20, scale: 0.95 },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        stiffness: 100,
        damping: 15,
      } as any,
    },
    exit: {
      opacity: 0,
      y: -20,
      scale: 0.95,
      transition: { duration: 0.2 },
    },
  };



  const pulseVariants = {
    pulse: {
      scale: [1, 1.05, 1],
      transition: {
        duration: 2,
        repeat: Infinity,
      },
    },
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-500';
    if (confidence >= 0.6) return 'bg-amber-500';
    return 'bg-red-500';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High Match';
    if (confidence >= 0.6) return 'Moderate Match';
    return 'Low Match';
  };

  return (
    <div className="w-full">
      {/* Loading State */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center py-12"
          >
            <motion.div
              variants={pulseVariants}
              animate="pulse"
              className="w-12 h-12 rounded-full border-4 border-amber-500 border-t-transparent"
            />
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="mt-4 text-slate-300 font-medium"
            >
              Analyzing sketch...
            </motion.p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error State */}
      <AnimatePresence>
        {error && !isLoading && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-red-900/20 border border-red-700 rounded-lg p-4 flex items-start gap-3"
          >
            <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="font-semibold text-red-300">Error</p>
              <p className="text-sm text-red-200">{error}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results Grid */}
      <AnimatePresence mode="wait">
        {matches.length > 0 && !isLoading && !error && (
          <motion.div
            key="results"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="space-y-4"
          >
            {/* Top Match - Featured */}
            {matches[0] && (
              <motion.div
                variants={itemVariants}
                className="relative overflow-hidden rounded-lg bg-gradient-to-r from-amber-600/20 to-amber-500/10 border-2 border-amber-500 p-6"
              >
                {/* Background glow */}
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-amber-500/10 to-transparent"
                  animate={{
                    opacity: [0.5, 0.8, 0.5],
                  }}
                  transition={{
                    duration: 3,
                    repeat: Infinity,
                  }}
                />

                <div className="relative z-10">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <motion.div
                  initial={{ scale: 0, rotate: -180 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{
                    stiffness: 100,
                    damping: 15,
                  } as any}
                      >
                        <CheckCircle className="w-6 h-6 text-amber-400" />
                      </motion.div>
                      <div>
                        <p className="text-sm font-semibold text-amber-200">Top Match</p>
                        <p className="text-2xl font-bold text-white">{matches[0].name}</p>
                      </div>
                    </div>
                    <motion.div
                      className="text-right"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: 0.3 }}
                    >
                      <p className="text-4xl font-bold text-amber-400">
                        {(matches[0].confidence * 100).toFixed(1)}%
                      </p>
                      <p className="text-xs text-amber-300">Confidence</p>
                    </motion.div>
                  </div>

                  {/* Confidence Bar */}
                  <motion.div
                    className="w-full h-2 bg-slate-700 rounded-full overflow-hidden"
                  >
                    <motion.div
                    initial={{ scaleX: 0, originX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{
                      stiffness: 80,
                      damping: 20,
                      delay: 0.2,
                    } as any}
                    className={`h-full ${getConfidenceColor(matches[0].confidence)}`}
                    style={{ width: `${matches[0].confidence * 100}%` }}
                    />
                  </motion.div>

                  <p className="text-xs text-amber-200 mt-2">
                    {getConfidenceLabel(matches[0].confidence)} • ID: {matches[0].id}
                  </p>
                </div>
              </motion.div>
            )}

            {/* Other Matches */}
            {matches.slice(1).map((match, idx) => (
              <motion.div
                key={match.id}
                variants={itemVariants}
                whileHover={{ scale: 1.02, x: 4 }}
                onClick={() => {
                  setSelectedMatch(match);
                  onMatchSelect?.(match);
                }}
                className="cursor-pointer rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors p-4 border border-slate-600 hover:border-amber-500/50"
              >
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <p className="font-semibold text-white text-sm">
                      #{idx + 2} - {match.name}
                    </p>
                    <p className="text-xs text-slate-400">ID: {match.id}</p>
                  </div>
                  <motion.div
                    className="text-right"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 + idx * 0.1 }}
                  >
                    <p className="font-bold text-amber-400">
                      {(match.confidence * 100).toFixed(1)}%
                    </p>
                  </motion.div>
                </div>

                {/* Confidence Bar */}
                <motion.div
                  className="w-full h-1.5 bg-slate-600 rounded-full overflow-hidden"
                >
                  <motion.div
                    initial={{ scaleX: 0, originX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{
                      stiffness: 80,
                      damping: 20,
                      delay: 0.3 + idx * 0.1,
                    } as any}
                    className={`h-full ${getConfidenceColor(match.confidence)}`}
                    style={{ width: `${match.confidence * 100}%` }}
                  />
                </motion.div>

                <p className="text-xs text-slate-400 mt-2">
                  {getConfidenceLabel(match.confidence)}
                </p>
              </motion.div>
            ))}

            {/* Trend Indicator */}
            {matches.length > 1 && (
              <motion.div
                variants={itemVariants}
                className="flex items-center gap-2 p-3 bg-slate-700/50 rounded-lg text-xs text-slate-300"
              >
                <TrendingUp className="w-4 h-4 text-blue-400" />
                <span>
                  Top match is {(
                    ((matches[0].confidence - matches[1].confidence) /
                      matches[1].confidence) *
                    100
                  ).toFixed(0)}
                  % more confident than #2
                </span>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty State */}
      <AnimatePresence>
        {matches.length === 0 && !isLoading && !error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="text-center py-12"
          >
            <p className="text-slate-400">
              Draw a sketch to see matching results
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Selected Match Detail */}
      <AnimatePresence>
        {selectedMatch && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="mt-6 p-6 bg-slate-700 rounded-lg border border-slate-600"
          >
            <div className="flex items-start justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">
                {selectedMatch.name}
              </h3>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setSelectedMatch(null)}
                className="text-slate-400 hover:text-white"
              >
                ✕
              </motion.button>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-slate-400 mb-1">Confidence Score</p>
                <p className="text-2xl font-bold text-amber-400">
                  {(selectedMatch.confidence * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-400 mb-1">Distance</p>
                <p className="text-2xl font-bold text-slate-300">
                  {selectedMatch.distance.toFixed(3)}
                </p>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-slate-600">
              <p className="text-xs text-slate-400 mb-2">Suspect ID</p>
              <p className="text-sm text-slate-200">{selectedMatch.id}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ResultsAnimation;
