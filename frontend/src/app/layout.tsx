import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Medical Investigation Intelligence",
  description: "AI-powered investigation report intelligence system",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
