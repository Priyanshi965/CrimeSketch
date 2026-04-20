import { z } from "zod";
import { publicProcedure, router } from "../_core/trpc";
import { TRPCError } from "@trpc/server";
import axios from "axios";

// ML Backend API base URL (will be set via environment variable)
const ML_API_URL = process.env.ML_API_URL || "http://localhost:8000";

// Schema for sketch upload
const SketchUploadSchema = z.object({
  imageData: z.string().describe("Base64 encoded image data"),
  topK: z.number().int().min(1).max(20).default(5),
});

// Schema for suspect filtering
const SuspectFilterSchema = z.object({
  city: z.string().optional(),
  crime_type: z.string().optional(),
  risk_level: z.enum(["Low", "Medium", "High"]).optional(),
  limit: z.number().int().min(1).max(100).default(20),
});

// Schema for adding suspect
const AddSuspectSchema = z.object({
  name: z.string().min(1),
  age: z.number().int().min(0).max(150),
  gender: z.enum(["Male", "Female", "Other"]),
  city: z.string().min(1),
  crime_type: z.string().min(1),
  risk_level: z.enum(["Low", "Medium", "High"]),
  imageData: z.string().describe("Base64 encoded image data"),
});

const TrainSchema = z.object({
  epochs: z.number().int().min(1).max(500).default(10),
  batchSize: z.number().int().min(1).max(512).default(16),
  learningRate: z.number().positive().max(1).default(0.0001),
  lossType: z.enum(["triplet", "contrastive"]).default("triplet"),
  targetAccuracy: z.number().min(0).max(1).default(0.94),
  topK: z.number().int().min(1).max(20).default(5),
  seed: z.number().int().min(0).max(1_000_000).default(42),
  reindexAfterTrain: z.boolean().default(true),
  maxTrainSamples: z.number().int().min(0).max(1_000_000).default(0),
  maxEvalSamples: z.number().int().min(0).max(1_000_000).default(0),
  gateMetric: z.enum(["accuracy", "rank1_accuracy"]).default("accuracy"),
  resumeFromCheckpoint: z.boolean().default(true),
  tripletMargin: z.number().positive().max(5).default(0.5),
  useAugmentation: z.boolean().default(true),
  batchHardTriplet: z.boolean().default(true),
  autoTune: z.boolean().default(true),
  maxAttempts: z.number().int().min(1).max(20).default(6),
});

export const mlRouter = router({
  // Predict: Upload sketch and get top-K matches
  predict: publicProcedure
    .input(SketchUploadSchema)
    .mutation(async ({ input }) => {
      try {
        // Call ML backend
        const response = await axios.post(`${ML_API_URL}/predict`, {
          image_data: input.imageData,
          top_k: input.topK,
        });
        const responseData = response.data ?? {};
        const rawMatches =
          responseData.top_k_matches ??
          responseData.matches ??
          responseData.topKMatches ??
          responseData.results ??
          [];

        const matches = (Array.isArray(rawMatches) ? rawMatches : []).map(
          (m: any) => ({
            ...m,
            image_url: m.suspect_id
              ? `${ML_API_URL}/image/${m.suspect_id}`
              : m.image_url,
          })
        );

        return {
          success: true,
          matches,
          pipeline_steps: responseData.pipeline_steps || [],
          preprocessing_info: responseData.preprocessing_info || {},
        };
      } catch (error: any) {
        console.error("ML Backend error:", error.message);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to process sketch. Please try again.",
        });
      }
    }),

  // Get suspects with filtering
  getSuspects: publicProcedure
    .input(SuspectFilterSchema)
    .query(async ({ input }) => {
      try {
        const params = new URLSearchParams();
        if (input.city) params.append("city", input.city);
        if (input.crime_type) params.append("crime_type", input.crime_type);
        if (input.risk_level) params.append("risk_level", input.risk_level);
        params.append("limit", input.limit.toString());

        const response = await axios.get(`${ML_API_URL}/suspects?${params}`);

        const suspects = (response.data.suspects || []).map((s: any) => ({
          ...s,
          image_url: s.id ? `${ML_API_URL}/image/${s.id}` : s.image_url,
        }));
        return {
          suspects,
          total: response.data.total || 0,
        };
      } catch (error: any) {
        console.error("Failed to fetch suspects:", error.message);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to fetch suspects.",
        });
      }
    }),

  // Add new suspect
  addSuspect: publicProcedure
    .input(AddSuspectSchema)
    .mutation(async ({ input }) => {
      try {
        const response = await axios.post(`${ML_API_URL}/add_suspect`, {
          name: input.name,
          age: input.age,
          gender: input.gender,
          city: input.city,
          crime_type: input.crime_type,
          risk_level: input.risk_level,
          image_data: input.imageData,
        });

        return {
          success: true,
          suspect_id: response.data.suspect_id,
          message: "Suspect added successfully",
        };
      } catch (error: any) {
        console.error("Failed to add suspect:", error.message);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to add suspect.",
        });
      }
    }),

  // Get system statistics
  getStats: publicProcedure.query(async () => {
    try {
      const response = await axios.get(`${ML_API_URL}/stats`);

      const datasetStats = response.data.dataset_stats || {};
      return {
        total_suspects: response.data.total_suspects || 0,
        total_embeddings: response.data.total_embeddings || 0,
        avg_confidence: response.data.avg_confidence || response.data.avg_match_confidence || 0,
        dataset1_sketches: datasetStats.dataset1_sketches || response.data.dataset1_sketches || 0,
        dataset1_photos: datasetStats.dataset1_photos || response.data.dataset1_photos || 0,
        dataset2_sketches: datasetStats.dataset2_sketches || response.data.dataset2_sketches || 0,
        dataset2_photos: datasetStats.dataset2_photos || response.data.dataset2_photos || 0,
        dataset3_faces: datasetStats.dataset3_faces || response.data.dataset3_faces || 0,
      };
    } catch (error: any) {
      console.error("Failed to fetch stats:", error.message);
      // Return default stats if backend is unavailable
      return {
        total_suspects: 0,
        total_embeddings: 0,
        avg_confidence: 0,
        dataset1_sketches: 0,
        dataset1_photos: 0,
        dataset2_sketches: 0,
        dataset2_photos: 0,
        dataset3_faces: 0,
      };
    }
  }),

  // Re-index database
  reindex: publicProcedure.mutation(async () => {
    try {
      const response = await axios.post(`${ML_API_URL}/re_index`);

      return {
        success: true,
        message: response.data.message || "Re-indexing completed",
        indexed_count: response.data.indexed_count || 0,
      };
    } catch (error: any) {
      console.error("Failed to re-index:", error.message);
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: "Failed to re-index database.",
      });
    }
  }),

  // Start model training
  train: publicProcedure
    .input(TrainSchema)
    .mutation(async ({ input }) => {
      try {
        const response = await axios.post(`${ML_API_URL}/train`, {
          epochs: input.epochs,
          batch_size: input.batchSize,
          learning_rate: input.learningRate,
          loss_type: input.lossType,
          target_accuracy: input.targetAccuracy,
          top_k: input.topK,
          seed: input.seed,
          reindex_after_train: input.reindexAfterTrain,
          max_train_samples: input.maxTrainSamples,
          max_eval_samples: input.maxEvalSamples,
          gate_metric: input.gateMetric,
          resume_from_checkpoint: input.resumeFromCheckpoint,
          triplet_margin: input.tripletMargin,
          use_augmentation: input.useAugmentation,
          batch_hard_triplet: input.batchHardTriplet,
          auto_tune: input.autoTune,
          max_attempts: input.maxAttempts,
        });

        return {
          success: true,
          status: response.data.status || "started",
          message: response.data.message || "Training started",
          config: response.data.config || null,
        };
      } catch (error: any) {
        console.error("Failed to start training:", error.message);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to start training.",
        });
      }
    }),

  // Training status
  trainStatus: publicProcedure.query(async () => {
    try {
      const response = await axios.get(`${ML_API_URL}/train/status`);
      return response.data;
    } catch (error: any) {
      console.error("Failed to fetch training status:", error.message);
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: "Failed to fetch training status.",
      });
    }
  }),

  // Get search history
  getSearchHistory: publicProcedure.query(async () => {
    try {
      const response = await axios.get(`${ML_API_URL}/search_history`);

      return response.data.history || [];
    } catch (error: any) {
      console.error("Failed to fetch search history:", error.message);
      return [];
    }
  }),

  // Health check
  health: publicProcedure.query(async () => {
    try {
      const response = await axios.get(`${ML_API_URL}/health`);
      return {
        status: "ok",
        backend_status: response.data.status || "unknown",
      };
    } catch (error) {
      return {
        status: "error",
        backend_status: "offline",
      };
    }
  }),
});
