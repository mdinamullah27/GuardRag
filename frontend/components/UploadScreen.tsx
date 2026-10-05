"use client";

import { ChangeEvent, DragEvent, useRef, useState } from "react";

interface Props {
  uploading: boolean;
  error: string | null;
  onFile: (file: File) => void;
}

const STEPS = ["Extracting text", "Generating embeddings", "Saving to vector database"];

export default function UploadScreen({ uploading, error, onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const pick = (file: File | null | undefined) => {
    if (file && !uploading) onFile(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    pick(e.dataTransfer.files?.[0]);
  };

  const onChange = (e: ChangeEvent<HTMLInputElement>) => pick(e.target.files?.[0]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-slate-50 px-4 py-10">
      <div className="w-full max-w-xl text-center">
        <div className="mb-1 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
          <span className="text-indigo-600">✦</span> GuardRAG
        </div>
        <p className="mb-8 text-sm text-slate-500 sm:text-base">Read smarter with AI</p>

        <div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
          <div
            role="button"
            tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`flex min-h-56 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 text-center transition ${
              dragging
                ? "border-indigo-400 bg-indigo-50"
                : "border-slate-300 hover:border-indigo-300 hover:bg-slate-50"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept="application/pdf,.pdf"
              className="hidden"
              onChange={onChange}
            />
            <svg viewBox="0 0 24 24" className="h-10 w-10 text-slate-300" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 16V4m0 0L8 8m4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" />
            </svg>
            <p className="text-sm font-medium text-slate-700 sm:text-base">Drop your PDF here</p>
            <p className="text-sm text-indigo-600 underline-offset-2 hover:underline">or Browse files</p>
            <p className="text-xs text-slate-400">Maximum size: 20 MB</p>
          </div>

          {error && (
            <p className="mt-4 rounded-lg bg-red-50 px-4 py-3 text-sm font-medium text-red-600">
              {error}
            </p>
          )}

          {uploading && (
            <div className="mt-5 rounded-xl border border-indigo-100 bg-indigo-50/60 px-5 py-4 text-left">
              <div className="flex items-center gap-2 text-sm font-semibold text-indigo-700">
                <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-500" />
                Indexing document…
              </div>
              <ul className="mt-3 space-y-1.5">
                {STEPS.map((step) => (
                  <li key={step} className="flex items-center gap-2 text-sm text-slate-600">
                    <span className="h-1.5 w-1.5 rounded-full bg-indigo-300" />
                    {step}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <p className="mt-6 text-xs text-slate-400">
          PDFs are chunked and embedded locally · stored in PostgreSQL + pgvector
        </p>
      </div>
    </main>
  );
}
