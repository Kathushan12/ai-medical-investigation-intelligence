export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000/api/v1";

export type Investigation = {
  id: string;
  original_filename: string;
  upload_status: string;
  ocr_status: string;
  embedding_status: string;
  processing_status: string;
  patient_name?: string | null;
  medical_condition?: string | null;
  hospital_location?: string | null;
  severity_level?: string | null;
  summary?: string | null;
  error_message?: string | null;
  created_at: string;
};

export type SearchResult = {
  investigation_id: string;
  chunk_id: string;
  score: number;
  content: string;
  original_filename: string;
  patient_name?: string | null;
  medical_condition?: string | null;
  hospital_location?: string | null;
  severity_level?: string | null;
};

export async function listInvestigations(): Promise<Investigation[]> {
  const res = await fetch(`${API_BASE}/investigations`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load investigations");
  return res.json();
}

export async function uploadInvestigation(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/investigations/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function hybridSearch(query: string): Promise<SearchResult[]> {
  const res = await fetch(`${API_BASE}/search/hybrid?q=${encodeURIComponent(query)}&limit=8`, { cache: "no-store" });
  if (!res.ok) throw new Error("Search failed");
  return res.json();
}

export async function askAssistant(question: string) {
  const res = await fetch(`${API_BASE}/assistant/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: 5 }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
