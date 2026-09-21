const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function generateVideo(params: {
  model?: string;
  prompt: string;
  image?: string;
  steps?: number;
  guidance_scale?: number;
  seed?: number | null;
}) {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "ti2v-5b", ...params }),
  });
  if (!res.ok) {
    const err = await res.text().catch(() => res.statusText);
    throw new Error(`API error (${res.status}): ${err}`);
  }
  return res.json();
}

export async function getJobStatus(jobId: string) {
  const res = await fetch(`${API_BASE}/api/status/${jobId}`);
  if (!res.ok) throw new Error(`Status check failed (${res.status})`);
  return res.json();
}

export async function getDownloadUrl(jobId: string) {
  const res = await fetch(`${API_BASE}/api/download/${jobId}`);
  if (!res.ok) throw new Error(`Download failed (${res.status})`);
  return res.json();
}
