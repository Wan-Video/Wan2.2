"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { generateVideo, getJobStatus } from "@/lib/api";

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

export default function ImageToVideoPage() {
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
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

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (f.size > MAX_FILE_SIZE) {
      setError("File too large. Max 10MB.");
      return;
    }
    if (!f.type.startsWith("image/")) {
      setError("Only image files allowed.");
      return;
    }
    setError(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

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
    if (!file) return;
    setLoading(true);
    setError(null);
    setOutputUrl(null);
    try {
      const base64 = await fileToBase64(file);
      const job = await generateVideo({ prompt, image: base64 });
      setStatus("queued");
      pollStatus(job.job_id);
    } catch (e: any) {
      setError(e.message);
      setLoading(false);
    }
  }

  function fileToBase64(f: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve((reader.result as string).split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(f);
    });
  }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Image to Video</h1>

      <div
        className="border-2 border-dashed rounded-lg p-8 text-center mb-4 cursor-pointer hover:border-blue-400 transition"
        onClick={() => fileRef.current?.click()}
      >
        {preview ? (
          <img src={preview} className="max-h-48 mx-auto rounded" alt="Preview" />
        ) : (
          <div>
            <p className="text-gray-400">Click to upload an image</p>
            <p className="text-gray-400 text-sm mt-1">Max 10MB, PNG/JPG</p>
          </div>
        )}
        <input ref={fileRef} type="file" accept="image/png,image/jpeg" className="hidden" onChange={handleFile} />
      </div>

      <textarea
        className="w-full p-3 border rounded-lg mb-4 h-24 resize-y"
        placeholder="Describe the motion or changes you want..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />

      <button
        className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
        onClick={handleGenerate}
        disabled={loading || !file}
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
