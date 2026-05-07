"use client";

import { useEffect, useState } from "react";
import { askAssistant, hybridSearch, Investigation, listInvestigations, SearchResult, uploadInvestigation } from "../lib/api";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [searchQuery, setSearchQuery] = useState("high risk diabetes Colombo");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [question, setQuestion] = useState("Show high-risk incidents in Colombo");
  const [assistantAnswer, setAssistantAnswer] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  async function refresh() {
    const data = await listInvestigations();
    setInvestigations(data);
  }

  useEffect(() => {
    refresh().catch(console.error);
    const timer = setInterval(() => refresh().catch(console.error), 5000);
    return () => clearInterval(timer);
  }, []);

  async function handleUpload() {
    if (!file) return;
    setLoading(true);
    setMessage("");
    try {
      await uploadInvestigation(file);
      setMessage("Uploaded successfully. Processing started.");
      setFile(null);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleSearch() {
    setLoading(true);
    try {
      setSearchResults(await hybridSearch(searchQuery));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleAssistant() {
    setLoading(true);
    try {
      const response = await askAssistant(question);
      setAssistantAnswer(response.answer);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Assistant failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen p-6 text-slate-900">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="rounded-3xl bg-white p-8 shadow-sm">
          <p className="text-sm font-semibold uppercase tracking-wide text-slate-500">AI Internship Task</p>
          <h1 className="mt-2 text-3xl font-bold">Medical Investigation Report Intelligence System</h1>
          <p className="mt-3 max-w-3xl text-slate-600">
            Upload medical investigation evidence, process it with AI OCR, extract structured fields,
            search with keyword/semantic/hybrid retrieval, and ask grounded RAG questions with citations.
          </p>
        </header>

        <section className="grid gap-6 md:grid-cols-2">
          <div className="rounded-3xl bg-white p-6 shadow-sm">
            <h2 className="text-xl font-semibold">1. Upload evidence</h2>
            <p className="mt-1 text-sm text-slate-600">Supported: PDF, PNG, JPG, JPEG, TXT</p>
            <input
              className="mt-4 block w-full rounded-xl border border-slate-200 p-3"
              type="file"
              accept=".pdf,.png,.jpg,.jpeg,.txt"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className="mt-4 rounded-xl bg-slate-900 px-5 py-3 font-medium text-white disabled:opacity-40"
            >
              Upload and process
            </button>
            {message && <p className="mt-3 text-sm text-slate-700">{message}</p>}
          </div>

          <div className="rounded-3xl bg-white p-6 shadow-sm">
            <h2 className="text-xl font-semibold">2. Ask AI assistant</h2>
            <textarea
              className="mt-4 h-24 w-full rounded-xl border border-slate-200 p-3"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
            <button onClick={handleAssistant} disabled={loading} className="mt-4 rounded-xl bg-slate-900 px-5 py-3 font-medium text-white">
              Ask with citations
            </button>
            {assistantAnswer && <pre className="mt-4 whitespace-pre-wrap rounded-xl bg-slate-100 p-4 text-sm">{assistantAnswer}</pre>}
          </div>
        </section>

        <section className="rounded-3xl bg-white p-6 shadow-sm">
          <h2 className="text-xl font-semibold">3. Hybrid search</h2>
          <div className="mt-4 flex gap-3">
            <input
              className="flex-1 rounded-xl border border-slate-200 p-3"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <button onClick={handleSearch} disabled={loading} className="rounded-xl bg-slate-900 px-5 py-3 font-medium text-white">
              Search
            </button>
          </div>
          <div className="mt-4 grid gap-3">
            {searchResults.map((result) => (
              <article key={result.chunk_id} className="rounded-2xl border border-slate-200 p-4">
                <div className="flex flex-wrap items-center gap-2 text-sm text-slate-600">
                  <span className="font-semibold text-slate-900">{result.original_filename}</span>
                  <span>Score: {result.score.toFixed(3)}</span>
                  {result.severity_level && <span>Severity: {result.severity_level}</span>}
                  {result.hospital_location && <span>Location: {result.hospital_location}</span>}
                </div>
                <p className="mt-2 text-sm text-slate-700">{result.content.slice(0, 500)}...</p>
              </article>
            ))}
          </div>
        </section>

        <section className="rounded-3xl bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">4. Investigation records</h2>
            <button onClick={refresh} className="rounded-xl border border-slate-200 px-4 py-2 text-sm">Refresh</button>
          </div>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-slate-500">
                <tr>
                  <th className="p-3">File</th>
                  <th className="p-3">Patient</th>
                  <th className="p-3">Condition</th>
                  <th className="p-3">Location</th>
                  <th className="p-3">Severity</th>
                  <th className="p-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {investigations.map((item) => (
                  <tr key={item.id} className="border-t border-slate-100">
                    <td className="p-3 font-medium">{item.original_filename}</td>
                    <td className="p-3">{item.patient_name || "-"}</td>
                    <td className="p-3">{item.medical_condition || "-"}</td>
                    <td className="p-3">{item.hospital_location || "-"}</td>
                    <td className="p-3">{item.severity_level || "-"}</td>
                    <td className="p-3">
                      <span className="rounded-full bg-slate-100 px-3 py-1 text-xs">{item.processing_status}</span>
                      {item.error_message && <p className="mt-1 text-xs text-red-600">{item.error_message}</p>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  );
}
