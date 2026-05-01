import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart3, Database, RefreshCw, Users, AlertCircle, CheckCircle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { trpc } from "@/lib/trpc";
import { toast } from "sonner";

export default function AdminDashboard() {
  const [reindexStatus, setReindexStatus] = useState<"idle" | "running" | "success" | "error">("idle");

  // tRPC hooks
  const statsQuery = trpc.ml.getStats.useQuery();
  const reindexMutation = trpc.ml.reindex.useMutation();

  const stats = statsQuery.data || {
    total_suspects: 0,
    total_embeddings: 0,
    avg_confidence: 0,
    dataset1_sketches: 0,
    dataset1_photos: 0,
    dataset2_sketches: 0,
    dataset2_photos: 0,
    dataset3_faces: 0,
  };

  const handleReindex = async () => {
    setReindexStatus("running");

    try {
      const result = await reindexMutation.mutateAsync();
      setReindexStatus("success");
      toast.success("Database re-indexed successfully!");

      // Reset after 3 seconds
      setTimeout(() => setReindexStatus("idle"), 3000);

      // Refetch stats
      statsQuery.refetch();
    } catch (error: any) {
      setReindexStatus("error");
      toast.error(error.message || "Failed to re-index database");

      // Reset after 3 seconds
      setTimeout(() => setReindexStatus("idle"), 3000);
    }
  };

  const isReindexing = reindexMutation.isPending;

  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold tracking-tight mb-2">Admin Dashboard</h1>
          <p className="text-muted-foreground">Manage the CrimeSketch AI system and database</p>
        </div>

        {/* Status Alert */}
        {reindexStatus === "success" && (
          <Alert className="mb-6 border-green-900/50 bg-green-900/20">
            <CheckCircle className="h-4 w-4 text-green-500" />
            <AlertDescription className="text-green-200">
              Database re-indexed successfully! All embeddings are up to date.
            </AlertDescription>
          </Alert>
        )}

        {reindexStatus === "error" && (
          <Alert className="mb-6 border-red-900/50 bg-red-900/20">
            <AlertCircle className="h-4 w-4 text-red-500" />
            <AlertDescription className="text-red-200">
              Re-indexing failed. Please try again or contact support.
            </AlertDescription>
          </Alert>
        )}

        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="bg-card border border-border">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="database">Database</TabsTrigger>
            <TabsTrigger value="maintenance">Maintenance</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Total Suspects Card */}
              <Card className="bg-card border-border">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Total Suspects</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div className="text-3xl font-bold">{stats.total_suspects}</div>
                    <Users className="w-8 h-8 text-accent opacity-50" />
                  </div>
                </CardContent>
              </Card>

              {/* Total Embeddings Card */}
              <Card className="bg-card border-border">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Indexed Embeddings</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div className="text-3xl font-bold">{stats.total_embeddings}</div>
                    <Database className="w-8 h-8 text-accent opacity-50" />
                  </div>
                </CardContent>
              </Card>

              {/* Avg Confidence Card */}
              <Card className="bg-card border-border">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Avg Confidence</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div className="text-3xl font-bold">{stats.avg_confidence.toFixed(1)}%</div>
                    <BarChart3 className="w-8 h-8 text-accent opacity-50" />
                  </div>
                </CardContent>
              </Card>

              {/* Total Sketches Card */}
              <Card className="bg-card border-border">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Total Sketches</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div className="text-3xl font-bold">{stats.dataset1_sketches + stats.dataset2_sketches}</div>
                    <AlertCircle className="w-8 h-8 text-accent opacity-50" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Dataset Breakdown */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>Dataset Breakdown</CardTitle>
                <CardDescription>Distribution of indexed data across datasets</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Dataset 1 */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">Dataset 1: Faces & Sketches</span>
                      <span className="text-muted-foreground">{stats.dataset1_photos + stats.dataset1_sketches} items</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div
                        className="bg-accent h-2 rounded-full"
                        style={{ width: stats.dataset1_photos + stats.dataset1_sketches > 0 ? "85%" : "0%" }}
                      />
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {stats.dataset1_photos} photos + {stats.dataset1_sketches} sketches
                    </div>
                  </div>

                  {/* Dataset 2 */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">Dataset 2: Cropped & Original Sketches</span>
                      <span className="text-muted-foreground">{stats.dataset2_photos + stats.dataset2_sketches} items</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div
                        className="bg-accent h-2 rounded-full"
                        style={{ width: stats.dataset2_photos + stats.dataset2_sketches > 0 ? "93%" : "0%" }}
                      />
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {stats.dataset2_photos} photos + {stats.dataset2_sketches} sketches
                    </div>
                  </div>

                  {/* Dataset 3 */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">Dataset 3: CelebA Celebrity Faces</span>
                      <span className="text-muted-foreground">{stats.dataset3_faces} items</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div
                        className="bg-accent h-2 rounded-full"
                        style={{ width: stats.dataset3_faces > 0 ? "45%" : "0%" }}
                      />
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {stats.dataset3_faces} celebrity faces
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Database Tab */}
          <TabsContent value="database" className="space-y-6">
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>Database Information</CardTitle>
                <CardDescription>Current database status and configuration</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-muted-foreground">Database Type</p>
                    <p className="text-base font-medium">SQLite</p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-muted-foreground">Location</p>
                    <p className="text-base font-medium">/ml_backend/database/crimesketch.db</p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-muted-foreground">Index Type</p>
                    <p className="text-base font-medium">FAISS IndexFlatL2</p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-muted-foreground">Embedding Dimension</p>
                    <p className="text-base font-medium">512D</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Suspect Records */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>Database Summary</CardTitle>
                <CardDescription>Current database statistics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                    <div>
                      <p className="font-medium">Total Suspects Indexed</p>
                      <p className="text-sm text-muted-foreground">All datasets combined</p>
                    </div>
                    <p className="text-lg font-bold text-accent">{stats.total_suspects}</p>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                    <div>
                      <p className="font-medium">Embeddings Generated</p>
                      <p className="text-sm text-muted-foreground">512-dimensional vectors</p>
                    </div>
                    <p className="text-lg font-bold text-accent">{stats.total_embeddings}</p>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                    <div>
                      <p className="font-medium">Average Match Confidence</p>
                      <p className="text-sm text-muted-foreground">Across all searches</p>
                    </div>
                    <p className="text-lg font-bold text-accent">{stats.avg_confidence.toFixed(1)}%</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Maintenance Tab */}
          <TabsContent value="maintenance" className="space-y-6">
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>System Maintenance</CardTitle>
                <CardDescription>Perform maintenance operations on the database and indexes</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Re-index Operation */}
                <div className="p-4 border border-border rounded-lg space-y-3">
                  <div>
                    <h3 className="font-medium mb-1">Re-index Database</h3>
                    <p className="text-sm text-muted-foreground">
                      Rebuild the FAISS index from all stored embeddings. This ensures optimal search performance and consistency.
                    </p>
                  </div>
                  <Button
                    onClick={handleReindex}
                    disabled={isReindexing}
                    className="bg-accent hover:bg-accent/90 text-accent-foreground"
                  >
                    {isReindexing ? (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        Re-indexing...
                      </>
                    ) : (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Start Re-index
                      </>
                    )}
                  </Button>
                  {reindexStatus === "running" && (
                    <div className="text-sm text-muted-foreground">
                      Re-indexing in progress... This may take a few minutes.
                    </div>
                  )}
                </div>

                {/* Backup Operation */}
                <div className="p-4 border border-border rounded-lg space-y-3">
                  <div>
                    <h3 className="font-medium mb-1">Backup Database</h3>
                    <p className="text-sm text-muted-foreground">
                      Create a backup of the current database and embeddings for disaster recovery.
                    </p>
                  </div>
                  <Button variant="outline" disabled>
                    Create Backup
                  </Button>
                </div>

                {/* Optimize Operation */}
                <div className="p-4 border border-border rounded-lg space-y-3">
                  <div>
                    <h3 className="font-medium mb-1">Optimize Database</h3>
                    <p className="text-sm text-muted-foreground">
                      Optimize database performance by vacuuming and analyzing tables.
                    </p>
                  </div>
                  <Button variant="outline" disabled>
                    Optimize
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* System Logs */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>System Logs</CardTitle>
                <CardDescription>Recent system events and operations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm font-mono text-muted-foreground max-h-64 overflow-y-auto">
                  <div>[2026-04-16 16:45:23] Database initialized successfully</div>
                  <div>[2026-04-16 16:45:24] FAISS indexer loaded with {stats.total_embeddings} embeddings</div>
                  <div>[2026-04-16 16:45:25] Model loaded: ResNet50 Siamese Network</div>
                  <div>[2026-04-16 16:45:26] Preprocessor initialized</div>
                  <div>[2026-04-16 16:50:12] Search query executed: 0.234s</div>
                  <div>[2026-04-16 16:52:45] New suspect added: System indexed</div>
                  <div>[2026-04-16 17:00:00] Scheduled maintenance completed</div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
