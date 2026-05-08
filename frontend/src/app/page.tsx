"use client";
import Image from "next/image";
import { useEffect, useMemo, useState } from "react";
import {
  askAssistant,
  hybridSearch,
  listInvestigations,
  uploadInvestigation,
} from "../lib/api";
import type { Investigation, SearchResult } from "../lib/api";

type AssistantCitation = {
  source_id?: string;
  original_filename?: string;
  excerpt?: string;
};

type AssistantApiResponse = {
  answer: string;
  citations?: AssistantCitation[];
};

function StatusBadge({ value }: { value?: string }) {
  const status = value || "unknown";

  const styles: Record<string, string> = {
    completed: "bg-emerald-50 text-emerald-700 ring-emerald-200",
    processing: "bg-blue-50 text-blue-700 ring-blue-200",
    pending: "bg-amber-50 text-amber-700 ring-amber-200",
    queued: "bg-purple-50 text-purple-700 ring-purple-200",
    failed: "bg-red-50 text-red-700 ring-red-200",
    unknown: "bg-slate-50 text-slate-700 ring-slate-200",
  };

  return (
    <span
      className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold capitalize ring-1 ${
        styles[status] || styles.unknown
      }`}
    >
      <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-current" />
      {status}
    </span>
  );
}

function StatCard({
  label,
  value,
  description,
}: {
  label: string;
  value: string | number;
  description: string;
}) {
  return (
    <div className="rounded-3xl border border-white/70 bg-white/80 p-5 shadow-sm backdrop-blur transition hover:-translate-y-0.5 hover:shadow-md">
      <p className="text-sm font-medium text-slate-500">{label}</p>
      <p className="mt-2 text-3xl font-bold text-slate-950">{value}</p>
      <p className="mt-1 text-sm text-slate-500">{description}</p>
    </div>
  );
}

function FeaturePill({ label }: { label: string }) {
  return (
    <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-700 ring-1 ring-slate-200">
      {label}
    </span>
  );
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [searchQuery, setSearchQuery] = useState("high-risk diabetes Colombo");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [question, setQuestion] = useState("Show high-risk incidents in Colombo");
  const [assistantAnswer, setAssistantAnswer] = useState("");
  const [assistantCitations, setAssistantCitations] = useState<
    AssistantCitation[]
  >([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  async function refresh() {
    try {
      const data = await listInvestigations();
      setInvestigations(data);
    } catch (error) {
      console.error(error);
      setMessage("Failed to load investigations. Please check backend connection.");
    }
  }

  useEffect(() => {
    refresh();

    const timer = setInterval(() => {
      refresh();
    }, 5000);

    return () => clearInterval(timer);
  }, []);

  const completedCount = useMemo(
    () =>
      investigations.filter((item) => item.processing_status === "completed")
        .length,
    [investigations]
  );

  const processingCount = useMemo(
    () =>
      investigations.filter(
        (item) =>
          item.processing_status === "processing" ||
          item.processing_status === "queued" ||
          item.processing_status === "pending"
      ).length,
    [investigations]
  );

  const failedCount = useMemo(
    () =>
      investigations.filter((item) => item.processing_status === "failed")
        .length,
    [investigations]
  );

  const reviewCount = useMemo(
    () => investigations.filter((item) => item.review_required).length,
    [investigations]
  );

  async function handleUpload() {
    if (!file) {
      setMessage("Please select a file before uploading.");
      return;
    }

    setLoading(true);
    setMessage("");

    try {
      await uploadInvestigation(file);
      setMessage("File uploaded successfully. Processing has started.");
      setFile(null);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSearch() {
    if (!searchQuery.trim()) {
      setMessage("Please enter a search query.");
      return;
    }

    setLoading(true);
    setMessage("");

    try {
      const results = await hybridSearch(searchQuery);
      setSearchResults(results);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Search failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleAssistant() {
    if (!question.trim()) {
      setMessage("Please enter a question.");
      return;
    }

    setLoading(true);
    setMessage("");
    setAssistantAnswer("");
    setAssistantCitations([]);

    try {
      const response = (await askAssistant(question)) as AssistantApiResponse;
      setAssistantAnswer(response.answer || "No supporting evidence found.");
      setAssistantCitations(response.citations || []);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Assistant failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-8">
        <section className="overflow-hidden rounded-[2rem] border border-white/70 bg-white/85 shadow-xl shadow-slate-200/70 backdrop-blur">
          <div className="relative p-8 sm:p-10 lg:p-12">
            <div className="absolute right-8 top-8 hidden rounded-full bg-blue-100 px-4 py-2 text-sm font-semibold text-blue-700 ring-1 ring-blue-200 lg:block">
              Medical Document Intelligence
            </div>

            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-3 rounded-full bg-white px-4 py-2 text-sm font-bold text-slate-900 shadow-sm ring-1 ring-slate-200">
                <Image
                  src="/logo-icon.png"
                  alt="Medical Investigation Intelligence Logo"
                  width={34}
                  height={34}
                  className="rounded-full"
                  priority
                />
                <span>Medical Investigation Intelligence Platform</span>
              </div>

              <h1 className="mt-6 text-4xl font-black tracking-tight text-slate-950 sm:text-5xl lg:text-6xl">
                Professional Medical Report Intelligence System
              </h1>

              <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-600">
                Upload investigation documents, extract medical details, track
                document quality, search reports, and ask evidence-based questions
                with clear citations.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <a
                  href="#upload"
                  className="rounded-2xl bg-slate-950 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-slate-300 transition hover:-translate-y-0.5 hover:bg-slate-800"
                >
                  Upload Evidence
                </a>

                <a
                  href="#assistant"
                  className="rounded-2xl border border-slate-200 bg-white px-6 py-3 text-sm font-bold text-slate-800 shadow-sm transition hover:-translate-y-0.5 hover:bg-slate-50"
                >
                  Ask Assistant
                </a>

                <a
                  href="http://localhost:8000/docs"
                  target="_blank"
                  className="rounded-2xl border border-blue-200 bg-blue-50 px-6 py-3 text-sm font-bold text-blue-700 shadow-sm transition hover:-translate-y-0.5 hover:bg-blue-100"
                >
                  API Docs
                </a>
              </div>
            </div>
          </div>
        </section>

        {message && (
          <div className="rounded-3xl border border-blue-100 bg-blue-50/90 p-4 text-sm font-medium text-blue-800 shadow-sm">
            {message}
          </div>
        )}

        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Total Records"
            value={investigations.length}
            description="Uploaded investigation files"
          />

          <StatCard
            label="Completed"
            value={completedCount}
            description="Ready for search and assistant use"
          />

          <StatCard
            label="Processing"
            value={processingCount}
            description="Currently being analyzed"
          />

          <StatCard
            label="Review Needed"
            value={reviewCount || failedCount}
            description="Low confidence or failed records"
          />
        </section>

        <section className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr]" id="upload">
          <div className="rounded-[2rem] border border-white/70 bg-white/90 p-6 shadow-lg shadow-slate-200/70 backdrop-blur">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-bold uppercase tracking-wide text-blue-600">
                  Upload Center
                </p>

                <h2 className="mt-2 text-2xl font-black text-slate-950">
                  Upload Investigation Evidence
                </h2>

                <p className="mt-2 text-sm leading-6 text-slate-600">
                  Add PDFs, scanned images, camera captures, or text reports. The
                  system will process them and show status, confidence, and review
                  information in the records table.
                </p>
              </div>

              <div className="rounded-2xl bg-blue-50 p-4 text-2xl ring-1 ring-blue-100">
                📄
              </div>
            </div>

            <div className="mt-6 rounded-3xl border-2 border-dashed border-slate-200 bg-slate-50/80 p-6 text-center transition hover:border-blue-300 hover:bg-blue-50/50">
              <input
                className="block w-full cursor-pointer rounded-2xl border border-slate-200 bg-white p-3 text-sm text-slate-700 file:mr-4 file:rounded-xl file:border-0 file:bg-slate-950 file:px-4 file:py-2 file:text-sm file:font-bold file:text-white"
                type="file"
                accept=".pdf,.png,.jpg,.jpeg,.txt"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />

              <p className="mt-3 text-xs text-slate-500">
                Supported formats: PDF, PNG, JPG, JPEG, TXT
              </p>

              {file && (
                <p className="mt-3 rounded-2xl bg-white px-4 py-3 text-sm font-semibold text-slate-700 shadow-sm">
                  Selected file: {file.name}
                </p>
              )}
            </div>

            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className="mt-5 w-full rounded-2xl bg-slate-950 px-5 py-4 text-sm font-black text-white shadow-lg shadow-slate-300 transition hover:-translate-y-0.5 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? "Processing..." : "Upload Document"}
            </button>
          </div>

          <div className="rounded-[2rem] border border-white/70 bg-white/90 p-6 shadow-lg shadow-slate-200/70 backdrop-blur">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-bold uppercase tracking-wide text-blue-600">
                  Quality Review
                </p>

                <h2 className="mt-2 text-2xl font-black text-slate-950">
                  Supported Evidence & Review Checks
                </h2>

                <p className="mt-2 text-sm leading-6 text-slate-600">
                  The interface highlights confidence, image quality, processing
                  progress, and human review requirements for each investigation.
                </p>
              </div>

              <div className="rounded-2xl bg-emerald-50 p-4 text-2xl ring-1 ring-emerald-100">
                ✅
              </div>
            </div>

            <div className="mt-6 grid gap-3">
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <p className="text-sm font-black text-slate-950">
                  Accepted document types
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <FeaturePill label="PDF" />
                  <FeaturePill label="Scanned Image" />
                  <FeaturePill label="Camera Image" />
                  <FeaturePill label="Text Report" />
                </div>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <p className="text-sm font-black text-slate-950">
                  Review indicators
                </p>
                <div className="mt-3 space-y-2 text-sm text-slate-600">
                  <p>• OCR confidence is shown under each file.</p>
                  <p>• Document quality score is shown when available.</p>
                  <p>• Low-confidence records are marked for review.</p>
                </div>
              </div>

              <div className="rounded-2xl border border-blue-100 bg-blue-50 p-4">
                <p className="text-sm font-black text-blue-800">
                  Current system status
                </p>
                <p className="mt-2 text-sm text-blue-700">
                  {investigations.length} total records · {completedCount} completed ·{" "}
                  {reviewCount} need review
                </p>
              </div>
            </div>
          </div>
        </section>

        <section
          className="grid gap-8 lg:grid-cols-[0.95fr_1.05fr]"
          id="assistant"
        >
          <div className="rounded-[2rem] border border-white/70 bg-white/90 p-6 shadow-lg shadow-slate-200/70 backdrop-blur">
            <p className="text-sm font-bold uppercase tracking-wide text-blue-600">
              Assistant
            </p>

            <h2 className="mt-2 text-2xl font-black text-slate-950">
              Ask Investigation Questions
            </h2>

            <p className="mt-2 text-sm leading-6 text-slate-600">
              Ask questions about uploaded reports. Answers are generated using
              available evidence and citations.
            </p>

            <textarea
              className="mt-5 h-32 w-full resize-none rounded-3xl border border-slate-200 bg-white p-4 text-sm text-slate-800 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask something like: Show high-risk incidents in Colombo"
            />

            <button
              onClick={handleAssistant}
              disabled={loading}
              className="mt-4 w-full rounded-2xl bg-blue-600 px-5 py-4 text-sm font-black text-white shadow-lg shadow-blue-200 transition hover:-translate-y-0.5 hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? "Thinking..." : "Ask with Citations"}
            </button>
          </div>

          <div className="rounded-[2rem] border border-white/70 bg-white/90 p-6 shadow-lg shadow-slate-200/70 backdrop-blur">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-bold uppercase tracking-wide text-blue-600">
                  Assistant Response
                </p>

                <h2 className="mt-2 text-2xl font-black text-slate-950">
                  Evidence-Based Answer
                </h2>
              </div>

              <div className="rounded-2xl bg-emerald-50 px-4 py-2 text-xs font-bold text-emerald-700 ring-1 ring-emerald-100">
                Citations Enabled
              </div>
            </div>

            <div className="mt-5 min-h-40 rounded-3xl bg-slate-950 p-5 text-sm leading-7 text-slate-100 shadow-inner">
              {assistantAnswer ? (
                <p className="whitespace-pre-wrap">{assistantAnswer}</p>
              ) : (
                <p className="text-slate-400">
                  Your assistant answer will appear here after asking a question.
                </p>
              )}
            </div>

            {assistantCitations.length > 0 && (
              <div className="mt-5 space-y-3">
                <h3 className="text-sm font-black text-slate-800">Citations</h3>

                {assistantCitations.map((citation, index) => (
                  <div
                    key={`${citation.source_id || index}-${index}`}
                    className="rounded-2xl border border-slate-200 bg-slate-50 p-4"
                  >
                    <p className="text-xs font-bold text-blue-700">
                      {citation.source_id || `S${index + 1}`} ·{" "}
                      {citation.original_filename || "Unknown source"}
                    </p>

                    {citation.excerpt && (
                      <p className="mt-2 text-sm leading-6 text-slate-600">
                        {citation.excerpt}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        <section className="rounded-[2rem] border border-white/70 bg-white/90 p-6 shadow-lg shadow-slate-200/70 backdrop-blur">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="text-sm font-bold uppercase tracking-wide text-blue-600">
                Search
              </p>

              <h2 className="mt-2 text-2xl font-black text-slate-950">
                Investigation Search
              </h2>

              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
                Search uploaded reports by condition, patient details, hospital,
                severity, or investigation content.
              </p>
            </div>

            <div className="flex w-full gap-3 lg:w-[520px]">
              <input
                className="min-w-0 flex-1 rounded-2xl border border-slate-200 bg-white p-4 text-sm outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Search reports..."
              />

              <button
                onClick={handleSearch}
                disabled={loading}
                className="rounded-2xl bg-slate-950 px-6 py-4 text-sm font-black text-white shadow-lg shadow-slate-300 transition hover:-translate-y-0.5 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Search
              </button>
            </div>
          </div>

          <div className="mt-6 grid gap-4">
            {searchResults.length === 0 ? (
              <div className="rounded-3xl border border-dashed border-slate-200 bg-slate-50 p-8 text-center text-sm text-slate-500">
                Search results will appear here.
              </div>
            ) : (
              searchResults.map((result) => (
                <article
                  key={result.chunk_id}
                  className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded-full bg-slate-950 px-3 py-1 text-xs font-bold text-white">
                      Score {Number(result.score || 0).toFixed(3)}
                    </span>

                    {result.severity_level && (
                      <span className="rounded-full bg-red-50 px-3 py-1 text-xs font-bold text-red-700 ring-1 ring-red-100">
                        {result.severity_level}
                      </span>
                    )}

                    {result.hospital_location && (
                      <span className="rounded-full bg-blue-50 px-3 py-1 text-xs font-bold text-blue-700 ring-1 ring-blue-100">
                        {result.hospital_location}
                      </span>
                    )}
                  </div>

                  <h3 className="mt-3 font-black text-slate-950">
                    {result.original_filename}
                  </h3>

                  <p className="mt-2 text-sm leading-7 text-slate-600">
                    {result.content.slice(0, 650)}
                    {result.content.length > 650 ? "..." : ""}
                  </p>
                </article>
              ))
            )}
          </div>
        </section>

        <section className="rounded-[2rem] border border-white/70 bg-white/90 p-6 shadow-lg shadow-slate-200/70 backdrop-blur">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-bold uppercase tracking-wide text-blue-600">
                Records
              </p>

              <h2 className="mt-2 text-2xl font-black text-slate-950">
                Investigation Records
              </h2>

              <p className="mt-2 text-sm text-slate-600">
                Track upload, OCR, embedding, processing, and review status.
              </p>
            </div>

            <button
              onClick={refresh}
              className="rounded-2xl border border-slate-200 bg-white px-5 py-3 text-sm font-bold text-slate-800 shadow-sm transition hover:-translate-y-0.5 hover:bg-slate-50"
            >
              Refresh Records
            </button>
          </div>

          <div className="mt-6 overflow-hidden rounded-3xl border border-slate-200 bg-white">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[1180px] text-left text-sm">
                <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="p-4">File</th>
                    <th className="p-4">Patient</th>
                    <th className="p-4">Condition</th>
                    <th className="p-4">Location</th>
                    <th className="p-4">Severity</th>
                    <th className="p-4">OCR</th>
                    <th className="p-4">Embedding</th>
                    <th className="p-4">Processing</th>
                    <th className="p-4">Review</th>
                  </tr>
                </thead>

                <tbody>
                  {investigations.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="p-8 text-center text-slate-500">
                        No investigations uploaded yet.
                      </td>
                    </tr>
                  ) : (
                    investigations.map((item) => (
                      <tr
                        key={item.id}
                        className="border-t border-slate-100 transition hover:bg-slate-50"
                      >
                        <td className="p-4">
                          <p className="font-bold text-slate-950">
                            {item.original_filename}
                          </p>

                          {item.document_quality_score !== null &&
                            item.document_quality_score !== undefined && (
                              <p className="mt-1 text-xs text-slate-500">
                                Quality:{" "}
                                {Number(item.document_quality_score).toFixed(2)}
                              </p>
                            )}

                          {item.ocr_confidence !== null &&
                            item.ocr_confidence !== undefined && (
                              <p className="mt-1 text-xs text-slate-500">
                                OCR confidence:{" "}
                                {Number(item.ocr_confidence).toFixed(2)}
                              </p>
                            )}

                          {item.error_message && (
                            <p className="mt-1 text-xs text-red-600">
                              {item.error_message}
                            </p>
                          )}
                        </td>

                        <td className="p-4 text-slate-600">
                          {item.patient_name || "-"}
                        </td>

                        <td className="p-4 text-slate-600">
                          {item.medical_condition || "-"}
                        </td>

                        <td className="p-4 text-slate-600">
                          {item.hospital_location || "-"}
                        </td>

                        <td className="p-4">
                          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-700">
                            {item.severity_level || "Unknown"}
                          </span>
                        </td>

                        <td className="p-4">
                          <StatusBadge value={item.ocr_status} />
                        </td>

                        <td className="p-4">
                          <StatusBadge value={item.embedding_status} />
                        </td>

                        <td className="p-4">
                          <StatusBadge value={item.processing_status} />
                        </td>

                        <td className="p-4">
                          {item.review_required ? (
                            <div>
                              <span className="rounded-full bg-red-50 px-3 py-1 text-xs font-bold text-red-700 ring-1 ring-red-100">
                                Review Required
                              </span>

                              {item.review_reason && (
                                <p className="mt-2 max-w-xs text-xs leading-5 text-red-600">
                                  {item.review_reason}
                                </p>
                              )}
                            </div>
                          ) : (
                            <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-bold text-emerald-700 ring-1 ring-emerald-100">
                              No Review
                            </span>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <footer className="pb-8 text-center text-sm text-slate-500">
          Built with FastAPI, PostgreSQL, pgvector, OpenAI, Next.js, TypeScript,
          Tailwind CSS, and Docker.
        </footer>
      </div>
    </main>
  );
}