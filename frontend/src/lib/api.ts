export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000/api/v1";

export type Investigation = {
  id: string;
  original_filename: string;
  mime_type?: string | null;

  upload_status: string;
  ocr_status: string;
  embedding_status: string;
  processing_status: string;

  patient_name?: string | null;
  medical_condition?: string | null;
  incident_date?: string | null;
  hospital_location?: string | null;
  severity_level?: string | null;
  doctor_notes?: string | null;
  lab_test_details?: string | null;
  summary?: string | null;

  ocr_confidence?: number | null;

  ocr_warnings?: string[] | null;
  ocr_metadata?: Record<string, unknown> | null;
  extraction_warnings?: string[] | null;

  document_quality_score?: number | null;
  blur_score?: number | null;

  review_required?: boolean | null;
  review_reason?: string | null;

  error_message?: string | null;

  created_at: string;
  updated_at?: string | null;
};

export type SearchResult = {
  investigation_id: string;
  chunk_id: string;
  chunk_index?: number;
  source_page?: number | null;

  score: number;
  keyword_score?: number | null;
  semantic_score?: number | null;

  content: string;
  original_filename: string;

  patient_name?: string | null;
  medical_condition?: string | null;
  hospital_location?: string | null;
  severity_level?: string | null;

  ocr_confidence?: number | null;
  review_required?: boolean | null;

  created_at?: string | null;
};

export type Citation = {
  source_id: string;
  investigation_id: string;
  chunk_id: string;
  original_filename: string;
  chunk_index: number;
  source_page?: number | null;
  excerpt: string;
};

export type AssistantResponse = {
  answer: string;
  citations: Citation[];
  retrieved_results: SearchResult[];
};

export type UploadResponse = {
  message: string;
  investigation: Investigation;
};

export async function listInvestigations(): Promise<Investigation[]> {
  const res = await fetch(`${API_BASE}/investigations`, {
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error("Failed to load investigations");
  }

  return res.json();
}

export async function uploadInvestigation(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/investigations/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}

export async function hybridSearch(query: string): Promise<SearchResult[]> {
  const res = await fetch(
    `${API_BASE}/search/hybrid?q=${encodeURIComponent(query)}&limit=8`,
    {
      cache: "no-store",
    }
  );

  if (!res.ok) {
    throw new Error("Search failed");
  }

  return res.json();
}

export async function askAssistant(
  question: string
): Promise<AssistantResponse> {
  const res = await fetch(`${API_BASE}/assistant/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      top_k: 5,
    }),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json();
}