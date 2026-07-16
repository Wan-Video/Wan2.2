import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Wan2.2 Studio",
  description: "AI video generation powered by Wan2.2",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <header className="border-b border-gray-200 bg-white">
          <nav className="max-w-4xl mx-auto flex justify-between items-center p-4">
            <Link href="/" className="text-xl font-bold text-gray-900">Wan2.2 Studio</Link>
            <div className="flex gap-4 text-sm">
              <Link href="/generate/text-to-video" className="text-gray-600 hover:text-blue-600">Text-to-Video</Link>
              <Link href="/generate/image-to-video" className="text-gray-600 hover:text-blue-600">Image-to-Video</Link>
              <Link href="/gallery" className="text-gray-600 hover:text-blue-600">Gallery</Link>
            </div>
          </nav>
        </header>
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
