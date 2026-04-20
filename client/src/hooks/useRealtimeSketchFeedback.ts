import { useEffect, useRef, useState, useCallback } from 'react';

export interface SketchFeedback {
  symmetry_score: number;
  completeness_score: number;
  clarity_score: number;
  overall_quality: number;
  issues: string[];
  suggestions: string[];
}

export interface Match {
  id: number;
  name: string;
  confidence: number;
  distance: number;
  image_url: string;
}

export interface RealtimeFeedbackResponse {
  status: 'success' | 'error' | 'debounced';
  quality_feedback?: SketchFeedback;
  matches?: Match[];
  edge_map?: string;
  attention_map?: string;
  preprocessed_image?: string;
  inference_time_ms?: number;
  message?: string;
}

interface UseRealtimeSketchFeedbackOptions {
  enabled: boolean;
  debounceMs?: number;
  apiUrl?: string;
}

export function useRealtimeSketchFeedback(options: UseRealtimeSketchFeedbackOptions) {
  const {
    enabled,
    debounceMs = 300,
    apiUrl = process.env.REACT_APP_ML_API_URL || 'http://localhost:8000'
  } = options;

  const [feedback, setFeedback] = useState<SketchFeedback | null>(null);
  const [matches, setMatches] = useState<Match[]>([]);
  const [edgeMap, setEdgeMap] = useState<string | null>(null);
  const [attentionMap, setAttentionMap] = useState<string | null>(null);
  const [preprocessedImage, setPreprocessedImage] = useState<string | null>(null);
  const [inferenceTime, setInferenceTime] = useState<number>(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastProcessTimeRef = useRef<number>(0);

  const processSketchFrame = useCallback(
    async (imageData: Blob) => {
      if (!enabled) return;

      // Debounce check
      const now = Date.now();
      if (now - lastProcessTimeRef.current < debounceMs) {
        return;
      }
      lastProcessTimeRef.current = now;

      try {
        setIsLoading(true);
        setError(null);

        // Convert blob to base64
        const reader = new FileReader();
        reader.onload = async (e) => {
          const base64Data = (e.target?.result as string)?.split(',')[1];

          try {
            const response = await fetch(`${apiUrl}/realtime-feedback`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                image_data: base64Data,
              }),
            });

            if (!response.ok) {
              throw new Error(`API error: ${response.statusText}`);
            }

            const data: RealtimeFeedbackResponse = await response.json();

            if (data.status === 'success') {
              if (data.quality_feedback) {
                setFeedback(data.quality_feedback);
              }
              if (data.matches) {
                setMatches(data.matches);
              }
              if (data.edge_map) {
                setEdgeMap(`data:image/png;base64,${data.edge_map}`);
              }
              if (data.attention_map) {
                setAttentionMap(`data:image/png;base64,${data.attention_map}`);
              }
              if (data.preprocessed_image) {
                setPreprocessedImage(`data:image/png;base64,${data.preprocessed_image}`);
              }
              if (data.inference_time_ms) {
                setInferenceTime(data.inference_time_ms);
              }
            } else if (data.status === 'error') {
              setError(data.message || 'Unknown error');
            }
          } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to process sketch');
          } finally {
            setIsLoading(false);
          }
        };
        reader.readAsDataURL(imageData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to process sketch');
        setIsLoading(false);
      }
    },
    [enabled, debounceMs, apiUrl]
  );

  const reset = useCallback(() => {
    setFeedback(null);
    setMatches([]);
    setEdgeMap(null);
    setAttentionMap(null);
    setPreprocessedImage(null);
    setInferenceTime(0);
    setError(null);
  }, []);

  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  return {
    feedback,
    matches,
    edgeMap,
    attentionMap,
    preprocessedImage,
    inferenceTime,
    isLoading,
    error,
    processSketchFrame,
    reset,
  };
}
