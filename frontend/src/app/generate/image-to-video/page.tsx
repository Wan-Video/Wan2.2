"use client";
import { useState, useRef } from "react";
import { generateVideo, getJobStatus } from "@/lib/api";

export default function ImageToVideoPage() {
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) { setFile(f); setPreview(URL.createObjectURL(f)); }
  }

  async function handleGenerate() {
    if (!file) return;
    setLoading(true);
    setOutputUrl(null);
    try {
      const base64 = await fileToBase64(file);
      const job = await generateVideo({ prompt, image: base64 });
      setStatus("queued");
      pollStatus(job.job_id);
    } catch (e: any) { alert(e.message); setLoading(false); }
  }

  function fileToBase64(f: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve((reader.result as string).split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(f);
    });
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
          clearInterval(interval); alert(job.error || "Failed");
        }
      } catch { clearInterval(interval); }
    }, 2000);
  }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Image to Video</h1>

      <div className="border-2 border-dashed rounded-lg p-8 text-center mb-4 cursor-pointer"
        onClick={() => fileRef.current?.click()}>
        {preview ? <img src={preview} className="max-h-48 mx-auto" /> :
          <p className="text-gray-400">Click to upload an image</p>}
        <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleFile} />
      </div>

      <textarea className="w-full p-3 border rounded-lg mb-4 h-24" placeholder="Describe motion..."
        value={prompt} onChange={(e) => setPrompt(e.target.value)} />

      <button className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium disabled:opacity-50"
        onClick={handleGenerate} disabled={loading || !file}>
        {loading ? "Generating..." : "Generate Video"}
      </button>

      {status && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <p>Status: <strong>{status}</strong></p>
          {status === "processing" && (
            <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
              <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${progress * 100}%` }} />
            </div>
          )}
          {outputUrl && <video className="w-full mt-4 rounded" controls src={outputUrl} />}
        </div>
      )}
    </div>
  );
}
