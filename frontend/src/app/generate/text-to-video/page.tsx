"use client";
import { useState } from "react";
import { generateVideo, getJobStatus } from "@/lib/api";

export default function TextToVideoPage() {
  const [prompt, setPrompt] = useState("");
  const [steps, setSteps] = useState(30);
  const [guidance, setGuidance] = useState(6);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleGenerate() {
    setLoading(true);
    setStatus("queued");
    setProgress(0);
    setOutputUrl(null);
    try {
      const job = await generateVideo({ prompt, steps, guidance_scale: guidance });
      setJobId(job.job_id);
      setStatus("queued");
      pollStatus(job.job_id);
    } catch (e: any) {
      setStatus("failed");
      alert(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function pollStatus(id: string) {
    const interval = setInterval(async () => {
      try {
        const job = await getJobStatus(id);
        setStatus(job.status);
        setProgress(job.progress);
        if (job.status === "completed") {
          setOutputUrl(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/download/${id}`);
          clearInterval(interval);
        } else if (job.status === "failed") {
          clearInterval(interval);
          alert(job.error || "Generation failed");
        }
      } catch { clearInterval(interval); }
    }, 2000);
  }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Text to Video</h1>

      <textarea
        className="w-full p-3 border rounded-lg mb-4 h-32"
        placeholder="Describe the video you want to generate..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block text-sm mb-1">Steps</label>
          <input type="number" className="w-full p-2 border rounded" value={steps}
            onChange={(e) => setSteps(+e.target.value)} />
        </div>
        <div>
          <label className="block text-sm mb-1">Guidance Scale</label>
          <input type="number" step="0.5" className="w-full p-2 border rounded" value={guidance}
            onChange={(e) => setGuidance(+e.target.value)} />
        </div>
      </div>

      <button
        className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium disabled:opacity-50"
        onClick={handleGenerate} disabled={loading || !prompt.trim()}>
        {loading ? "Generating..." : "Generate Video"}
      </button>

      {status && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <p className="text-sm mb-2">Status: <strong>{status}</strong></p>
          {status === "processing" && (
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${progress * 100}%` }} />
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
