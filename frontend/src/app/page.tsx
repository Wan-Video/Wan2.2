import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white">
      <main className="max-w-4xl mx-auto p-8 text-center pt-24">
        <h2 className="text-5xl font-bold mb-4">Wan2.2 Video Generation</h2>
        <p className="text-xl text-gray-300 mb-8">
          Open-source AI video generation. Powered by Wan2.2.
        </p>

        <div className="grid md:grid-cols-2 gap-6 max-w-2xl mx-auto">
          <Link
            href="/generate/text-to-video"
            className="p-8 bg-gray-800 rounded-xl hover:bg-gray-700 transition border border-gray-700"
          >
            <div className="text-4xl mb-3">🎬</div>
            <h3 className="text-xl font-semibold mb-2">Text to Video</h3>
            <p className="text-gray-400 text-sm">
              Generate videos from text prompts
            </p>
          </Link>
          <Link
            href="/generate/image-to-video"
            className="p-8 bg-gray-800 rounded-xl hover:bg-gray-700 transition border border-gray-700"
          >
            <div className="text-4xl mb-3">🖼️</div>
            <h3 className="text-xl font-semibold mb-2">Image to Video</h3>
            <p className="text-gray-400 text-sm">
              Animate your images with AI
            </p>
          </Link>
        </div>

        <div className="mt-12 text-sm text-gray-500">
          <p>Models: TI2V-5B | I2V-A14B | T2V-A14B</p>
          <p className="mt-1">Resolution: 720p @ 24fps</p>
        </div>
      </main>
    </div>
  );
}
