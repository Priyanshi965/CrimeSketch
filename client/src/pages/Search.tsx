import { useEffect, useMemo, useRef, useState } from "react";
import { trpc } from "@/lib/trpc";
import { toast } from "sonner";
import "./forensic-console.css";

type ViewKey = "match" | "suspects" | "analytics" | "training";

interface MatchResult {
  suspect_id?: number;
  id?: number;
  name?: string;
  city?: string;
  crime_type?: string;
  risk_level?: string;
  status?: string;
  image_url?: string;
  distance?: number;
  confidence?: number;
}

const PIPELINE = ["Input", "Preprocess", "ResNet50", "L2", "FAISS", "Result"];

export default function Search() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [activeView, setActiveView] = useState<ViewKey>("match");
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [topK, setTopK] = useState(5);
  const [activePipelineStep, setActivePipelineStep] = useState(0);
  const [results, setResults] = useState<MatchResult[]>([]);
  const [cityFilter, setCityFilter] = useState("");
  const [crimeFilter, setCrimeFilter] = useState("");
  const [riskFilter, setRiskFilter] = useState<"" | "Low" | "Medium" | "High">("");
  const [statusFilter, setStatusFilter] = useState("");
  const [searchText, setSearchText] = useState("");

  const [trainConfig, setTrainConfig] = useState({
    epochs: 24,
    batchSize: 16,
    learningRate: 0.0001,
    lossType: "triplet" as "triplet" | "contrastive",
    autoTune: true,
    maxAttempts: 6,
    targetAccuracy: 0.94,
  });

  const statsQuery = trpc.ml.getStats.useQuery();
  const suspectsQuery = trpc.ml.getSuspects.useQuery({
    city: cityFilter || undefined,
    crime_type: crimeFilter || undefined,
    risk_level: riskFilter || undefined,
    limit: 100,
  });
  const predictMutation = trpc.ml.predict.useMutation();
  const trainMutation = trpc.ml.train.useMutation();
  const trainStatusQuery = trpc.ml.trainStatus.useQuery(undefined, {
    refetchInterval: q => (q.state.data?.status === "running" ? 3000 : false),
  });

  useEffect(() => {
    if (!predictMutation.isPending) {
      setActivePipelineStep(results.length > 0 ? 5 : 0);
      return;
    }

    setActivePipelineStep(0);
    const timer = setInterval(() => {
      setActivePipelineStep(prev => (prev >= 5 ? 5 : prev + 1));
    }, 450);

    return () => clearInterval(timer);
  }, [predictMutation.isPending, results.length]);

  const topMatch = results[0];

  const filteredSuspects = useMemo(() => {
    const suspects = suspectsQuery.data?.suspects || [];
    if (!searchText.trim()) return suspects;
    const term = searchText.toLowerCase();
    return suspects.filter((s: any) => {
      const hay = `${s.name || ""} ${s.id || ""} ${s.city || ""} ${s.crime_type || ""}`.toLowerCase();
      return hay.includes(term);
    });
  }, [suspectsQuery.data?.suspects, searchText]);

  const confidencePct = ((topMatch?.confidence || 0) * 100).toFixed(1);

  const handleUpload = (file: File) => {
    const reader = new FileReader();
    reader.onload = e => {
      const data = e.target?.result as string;
      setUploadedImage(data);
      setResults([]);
    };
    reader.readAsDataURL(file);
  };

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    handleUpload(file);
  };

  const runMatch = async () => {
    if (!uploadedImage) {
      toast.error("Upload a sketch or photo first.");
      return;
    }

    try {
      const base64 = uploadedImage.split(",")[1] || uploadedImage;
      const response = await predictMutation.mutateAsync({ imageData: base64, topK });
      const found = (response.matches || []) as MatchResult[];
      setResults(found);
      toast.success(found.length > 0 ? `Found ${found.length} matches.` : "No matches found.");
    } catch (error: any) {
      toast.error(error?.message || "Prediction failed.");
    }
  };

  const startTraining = async () => {
    try {
      const res = await trainMutation.mutateAsync({
        epochs: trainConfig.epochs,
        batchSize: trainConfig.batchSize,
        learningRate: trainConfig.learningRate,
        lossType: trainConfig.lossType,
        targetAccuracy: trainConfig.targetAccuracy,
        topK,
        autoTune: trainConfig.autoTune,
        maxAttempts: trainConfig.maxAttempts,
      });

      toast.success(res.message || "Training started.");
      void trainStatusQuery.refetch();
      setActiveView("training");
    } catch (error: any) {
      toast.error(error?.message || "Failed to start training.");
    }
  };

  const renderMatch = () => (
    <div className="fc-layout">
      <aside className="fc-left">
        <h3 className="fc-section-title">AI Pipeline</h3>
        <div className="fc-pipeline">
          {PIPELINE.map((step, index) => (
            <div className={`fc-pipe-node ${index <= activePipelineStep ? "active" : ""}`} key={step}>
              {step}
            </div>
          ))}
        </div>

        <div
          className="fc-drop"
          onClick={() => fileRef.current?.click()}
          onDrop={e => {
            e.preventDefault();
            const file = e.dataTransfer.files?.[0];
            if (file) handleUpload(file);
          }}
          onDragOver={e => e.preventDefault()}
        >
          <p className="fc-drop-title">Drop Forensic Sketch</p>
          <p className="fc-drop-sub">PNG/JPG up to 10MB</p>
        </div>

        {uploadedImage && <img className="fc-preview" src={uploadedImage} alt="Uploaded sketch" />}

        <div className="fc-row">
          <label>TOP-K</label>
          <input
            type="range"
            min={1}
            max={10}
            value={topK}
            onChange={e => setTopK(Number(e.target.value))}
          />
          <span>{topK}</span>
        </div>

        <button className="fc-btn primary" onClick={runMatch} disabled={predictMutation.isPending}>
          {predictMutation.isPending ? "Running..." : "Run Match"}
        </button>
        <button className="fc-btn" onClick={() => fileRef.current?.click()}>
          Upload File
        </button>
        <input ref={fileRef} type="file" accept="image/*" onChange={onFileInput} hidden />
      </aside>

      <section className="fc-right">
        {!topMatch ? (
          <div className="fc-empty">Upload sketch and run match to see results</div>
        ) : (
          <>
            <div className="fc-top-card">
              <div>
                <div className="fc-muted">Top Match</div>
                <h2>{topMatch.name || "Unknown"}</h2>
                <div className="fc-muted">
                  {(topMatch.city || "Unknown city")} · {(topMatch.crime_type || "Unknown crime")}
                </div>
              </div>
              <div className="fc-score">{confidencePct}%</div>
            </div>

            <div className="fc-grid-results">
              {results.map((item, idx) => (
                <article className="fc-card" key={`${item.suspect_id || item.id || idx}`}>
                  {item.image_url ? <img src={item.image_url} alt={item.name || "match"} /> : <div className="fc-ph">No image</div>}
                  <div className="fc-card-body">
                    <strong>{item.name || "Unknown"}</strong>
                    <span>{(((item.confidence || 0) * 100) as number).toFixed(1)}%</span>
                  </div>
                </article>
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  );

  const renderSuspects = () => (
    <div className="fc-table-wrap">
      <div className="fc-filters">
        <input
          placeholder="Search name, ID, city"
          value={searchText}
          onChange={e => setSearchText(e.target.value)}
        />
        <select value={cityFilter} onChange={e => setCityFilter(e.target.value)}>
          <option value="">All Cities</option>
          <option>Mumbai</option>
          <option>Delhi</option>
          <option>Pune</option>
          <option>Bengaluru</option>
        </select>
        <select value={crimeFilter} onChange={e => setCrimeFilter(e.target.value)}>
          <option value="">All Crimes</option>
          <option>Theft</option>
          <option>Fraud</option>
          <option>Assault</option>
          <option>Murder</option>
        </select>
        <select value={riskFilter} onChange={e => setRiskFilter(e.target.value as "" | "Low" | "Medium" | "High")}>
          <option value="">All Risk</option>
          <option>High</option>
          <option>Medium</option>
          <option>Low</option>
        </select>
        <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
          <option value="">All Status</option>
          <option>Active</option>
          <option>Under Investigation</option>
          <option>Closed</option>
        </select>
      </div>

      <table className="fc-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>City</th>
            <th>Crime</th>
            <th>Risk</th>
          </tr>
        </thead>
        <tbody>
          {filteredSuspects
            .filter((s: any) => !statusFilter || (s.status || "").toLowerCase() === statusFilter.toLowerCase())
            .map((suspect: any) => (
              <tr key={suspect.id}>
                <td>{suspect.id}</td>
                <td>{suspect.name}</td>
                <td>{suspect.city || "-"}</td>
                <td>{suspect.crime_type || "-"}</td>
                <td>{suspect.risk_level || "-"}</td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );

  const renderAnalytics = () => {
    const stats = statsQuery.data;
    return (
      <div className="fc-analytics">
        <div className="fc-metric">
          <div>Target Accuracy</div>
          <strong>94%</strong>
        </div>
        <div className="fc-metric">
          <div>Indexed Faces</div>
          <strong>{stats?.total_embeddings || 0}</strong>
        </div>
        <div className="fc-metric">
          <div>Total Suspects</div>
          <strong>{stats?.total_suspects || 0}</strong>
        </div>
        <div className="fc-metric">
          <div>Avg Confidence</div>
          <strong>{(stats?.avg_confidence || 0).toFixed(1)}%</strong>
        </div>
      </div>
    );
  };

  const trainStatus = trainStatusQuery.data;
  const gate = trainStatus?.last_report?.gate;

  const renderTraining = () => (
    <div className="fc-training">
      <div className="fc-train-grid">
        <label>
          Epochs
          <input
            type="number"
            min={1}
            max={500}
            value={trainConfig.epochs}
            onChange={e => setTrainConfig(prev => ({ ...prev, epochs: Number(e.target.value) }))}
          />
        </label>
        <label>
          Batch Size
          <input
            type="number"
            min={1}
            max={512}
            value={trainConfig.batchSize}
            onChange={e => setTrainConfig(prev => ({ ...prev, batchSize: Number(e.target.value) }))}
          />
        </label>
        <label>
          Learning Rate
          <input
            type="number"
            step="0.00001"
            min={0.00001}
            value={trainConfig.learningRate}
            onChange={e => setTrainConfig(prev => ({ ...prev, learningRate: Number(e.target.value) }))}
          />
        </label>
        <label>
          Loss Type
          <select
            value={trainConfig.lossType}
            onChange={e => setTrainConfig(prev => ({ ...prev, lossType: e.target.value as "triplet" | "contrastive" }))}
          >
            <option value="triplet">Triplet</option>
            <option value="contrastive">Contrastive</option>
          </select>
        </label>
        <label>
          Max Attempts
          <input
            type="number"
            min={1}
            max={20}
            value={trainConfig.maxAttempts}
            onChange={e => setTrainConfig(prev => ({ ...prev, maxAttempts: Number(e.target.value) }))}
          />
        </label>
      </div>

      <label className="fc-check">
        <input
          type="checkbox"
          checked={trainConfig.autoTune}
          onChange={e => setTrainConfig(prev => ({ ...prev, autoTune: e.target.checked }))}
        />
        Auto-tune until reaching 94% target or attempts are exhausted
      </label>

      <button className="fc-btn primary" onClick={startTraining} disabled={trainMutation.isPending || trainStatus?.status === "running"}>
        {trainStatus?.status === "running" ? "Training in progress..." : "Start Training"}
      </button>

      <div className="fc-status-box">
        <p>Job Status: <strong>{trainStatus?.status || "idle"}</strong></p>
        {gate && (
          <p>
            Gate: {(gate.actual * 100).toFixed(2)}% / {(gate.target * 100).toFixed(2)}% ({gate.passed ? "PASSED" : "FAILED"})
          </p>
        )}
      </div>
    </div>
  );

  return (
    <div className="fc-shell">
      <header className="fc-topbar">
        <div>
          <h1>CID FACEMATCH</h1>
          <span>Forensic Intelligence System</span>
        </div>
        <div className="fc-badges">
          <span>FAISS ONLINE</span>
          <span>DB CONNECTED</span>
          <span>API READY</span>
        </div>
      </header>

      <main className="fc-main">
        <aside className="fc-sidebar">
          <button className={activeView === "match" ? "active" : ""} onClick={() => setActiveView("match")}>Face Match</button>
          <button className={activeView === "suspects" ? "active" : ""} onClick={() => setActiveView("suspects")}>Suspects</button>
          <button className={activeView === "analytics" ? "active" : ""} onClick={() => setActiveView("analytics")}>Analytics</button>
          <button className={activeView === "training" ? "active" : ""} onClick={() => setActiveView("training")}>Training</button>
        </aside>

        <section className="fc-content">
          {activeView === "match" && renderMatch()}
          {activeView === "suspects" && renderSuspects()}
          {activeView === "analytics" && renderAnalytics()}
          {activeView === "training" && renderTraining()}
        </section>
      </main>
    </div>
  );
}