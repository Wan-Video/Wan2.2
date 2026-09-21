"use client";
import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Job {
  job_id: string;
  status: string;
  output_url?: string;
  created_at: string;
  prompt: string;
}

export default function GalleryPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchJobs() {
      try {
        const res = await fetch(`${API_BASE}/api/gallery`);
        if (res.ok) {
          setJobs(await res.json());
        }
      } catch {
        // API might not have gallery endpoint yet
      } finally {
        setLoading(false);
      }
    }
    fetchJobs();
  }, []);

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Gallery</h1>
      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : jobs.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <div className="text-4xl mb-3">🎥</div>
          <p>No videos generated yet.</p>
          <p className="text-sm mt-1">Go to Text-to-Video or Image-to-Video to create one.</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {jobs.map((job) => (
            <div key={job.job_id} className="border rounded-lg overflow-hidden">
              {job.status === "completed" && job.output_url ? (
                <video controls className="w-full" src={`${API_BASE}${job.output_url}`} />
              ) : (
                <div className="bg-gray-100 h-32 flex items-center justify-center text-gray-400">
                  {job.status}
                </div>
              )}
              <div className="p-2 text-xs text-gray-500 truncate">{job.prompt}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
