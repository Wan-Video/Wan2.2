"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { generateVideo, getJobStatus } from "@/lib/api";

const MODELS = [
  { value: "ti2v-5b", label: "TI2V-5B (Fast, 720p)", vram: "8GB" },
  { value: "i2v-14b", label: "I2V-A14B (Quality, 720p)", vram: "16GB" },
  { value: "t2v-14b", label: "T2V-A14B (Text-only, 720p)", vram: "24GB" },
];

export default function TextToVideoPage() {
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState("ti2v-5b");
  const [steps, setSteps] = useState(30);
  const [guidance, setGuidance] = useState(6);
  const [status, setStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => clearPolling();
  }, [clearPolling]);

  async function pollStatus(id: string) {
    clearPolling();
    intervalRef.current = setInterval(async () => {
      try {
        const job = await getJobStatus(id);
        setStatus(job.status);
        setProgress(job.progress);
        if (job.status === "completed") {
          setOutputUrl(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/download/${id}`);
          setLoading(false);
          clearPolling();
        } else if (job.status === "failed") {
          setError(job.error || "Generation failed");
          setLoading(false);
          clearPolling();
        }
      } catch {
        clearPolling();
        setLoading(false);
      }
    }, 2000);
  }

  async function handleGenerate() {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    setStatus("queued");
    setProgress(0);
    setOutputUrl(null);
    try {
      const job = await generateVideo({ model, prompt, steps, guidance_scale: guidance });
      setStatus("queued");
      pollStatus(job.job_id);
    } catch (e: any) {
      setStatus("failed");
      setError(e.message);
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Text to Video</h1>

      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Model</label>
        <select
          className="w-full p-2 border rounded"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          {MODELS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label} ({m.vram})
            </option>
          ))}
        </select>
      </div>

      <textarea
        className="w-full p-3 border rounded-lg mb-4 h-32 resize-y"
        placeholder="Describe the video you want to generate..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1">Steps</label>
          <input
            type="number" min={10} max={100}
            className="w-full p-2 border rounded"
            value={steps}
            onChange={(e) => setSteps(+e.target.value)}
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Guidance Scale</label>
          <input
            type="number" min={1} max={20} step={0.5}
            className="w-full p-2 border rounded"
            value={guidance}
            onChange={(e) => setGuidance(+e.target.value)}
          />
        </div>
      </div>

      <button
        className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
        onClick={handleGenerate}
        disabled={loading || !prompt.trim()}
      >
        {loading ? "Generating..." : "Generate Video"}
      </button>

      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
          {error}
        </div>
      )}

      {status && !error && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <p className="text-sm mb-2">
            Status: <strong className="capitalize">{status}</strong>
            {status === "processing" && ` (${Math.round(progress * 100)}%)`}
          </p>
          {status === "processing" && (
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${Math.max(progress * 100, 5)}%` }}
              />
            </div>
          )}
          {outputUrl && (
            <video className="w-full mt-4 rounded" controls src={outputUrl} />
          )}
        </div>
      )}
    </div>
  );
}
