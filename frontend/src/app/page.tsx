import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white">
      <header className="border-b border-gray-700 p-4">
        <nav className="max-w-4xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold">Wan2.2 Studio</h1>
          <div className="flex gap-4">
            <Link href="/generate/text-to-video" className="hover:text-blue-400">Text-to-Video</Link>
            <Link href="/generate/image-to-video" className="hover:text-blue-400">Image-to-Video</Link>
            <Link href="/gallery" className="hover:text-blue-400">Gallery</Link>
          </div>
        </nav>
      </header>

      <main className="max-w-4xl mx-auto p-8 text-center pt-24">
        <h2 className="text-5xl font-bold mb-4">Wan2.2 Video Generation</h2>
        <p className="text-xl text-gray-300 mb-8">Open-source AI video generation powered by Wan2.2</p>

        <div className="grid grid-cols-2 gap-6 max-w-2xl mx-auto">
          <Link href="/generate/text-to-video"
            className="p-8 bg-gray-800 rounded-xl hover:bg-gray-700 transition">
            <h3 className="text-xl font-semibold mb-2">Text to Video</h3>
            <p className="text-gray-400 text-sm">Generate videos from text prompts</p>
          </Link>
          <Link href="/generate/image-to-video"
            className="p-8 bg-gray-800 rounded-xl hover:bg-gray-700 transition">
            <h3 className="text-xl font-semibold mb-2">Image to Video</h3>
            <p className="text-gray-400 text-sm">Animate your images with AI</p>
          </Link>
        </div>
      </main>
    </div>
  );
}
