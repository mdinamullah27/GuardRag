"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import type { ChatMessage } from "@/lib/types";

interface Props {
  open: boolean;
  onToggle: () => void;
  messages: ChatMessage[];
  sending: boolean;
  selectedText: string | null;
  onSend: (question: string) => void;
  onNavigatePage: (page: number) => void;
  onDismissSelection: () => void;
}

const SUGGESTIONS = [
  "Summarize this document",
  "What is the main contribution?",
  "List the key findings",
];

function ModeChips({ mode, completeness }: { mode?: string; completeness?: string | null }) {
  return (
    <>
      {mode && mode !== "NORMAL" && (
        <span
          title="Retrieval strategy used for this answer"
          className="rounded-full bg-indigo-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-indigo-600"
        >
          {mode.toLowerCase()}
        </span>
      )}
      {completeness === "complete" && (
        <span
          title="The whole document was covered by this answer"
          className="rounded-full bg-sky-50 px-2 py-0.5 text-[10px] font-medium text-sky-700"
        >
          full document
        </span>
      )}
      {completeness === "partial" && (
        <span
          title="Only part of the document could be covered"
          className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-medium text-amber-700"
        >
          partial coverage
        </span>
      )}
    </>
  );
}

export default function FloatingChat({
  open,
  onToggle,
  messages,
  sending,
  selectedText,
  onSend,
  onNavigatePage,
  onDismissSelection,
}: Props) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending, open]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const question = input.trim();
    if (!question || sending) return;
    setInput("");
    onSend(question);
  };

  const ask = (question: string) => {
    if (sending) return;
    onSend(question);
  };

  return (
    <>
      {selectedText && open && (
        <div className="fixed bottom-24 left-1/2 z-40 w-[min(24rem,calc(100vw-2rem))] -translate-x-1/2 rounded-xl border border-indigo-200 bg-white px-4 py-3 shadow-lg sm:left-auto sm:right-6 sm:translate-x-0">
          <p className="line-clamp-3 text-sm italic text-slate-600">“{selectedText}”</p>
          <div className="mt-2 flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={onDismissSelection}
              className="text-xs text-slate-400 hover:text-slate-600"
            >
              Clear
            </button>
            <button
              type="button"
              onClick={() => ask(`Explain this passage: "${selectedText}"`)}
              className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-indigo-500"
            >
              ✨ Ask GuardRAG
            </button>
          </div>
        </div>
      )}

      {open ? (
        <section
          aria-label="GuardRAG chat"
          className="fixed inset-x-0 bottom-0 z-40 flex h-[85vh] flex-col rounded-t-2xl border border-slate-200 bg-white shadow-2xl sm:inset-x-auto sm:right-4 sm:bottom-4 sm:h-[min(36rem,calc(100vh-6rem))] sm:w-[24rem] sm:rounded-2xl"
        >
          <header className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
              <span className="text-indigo-600">✦</span> GuardRAG Chat
              <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-600">
                grounded
              </span>
            </div>
            <button
              type="button"
              onClick={onToggle}
              aria-label="Close chat"
              className="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="h-4 w-4">
                <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" />
              </svg>
            </button>
          </header>

          <div ref={scrollRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-4">
            {messages.length === 0 && !sending && (
              <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
                <span className="text-3xl text-indigo-500">✦</span>
                <p className="text-sm font-medium text-slate-600">Ask anything about this document</p>
                <div className="flex flex-col gap-1.5">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => ask(s)}
                      className="rounded-full border border-slate-200 px-3 py-1.5 text-xs text-slate-600 transition hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-700"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg) =>
              msg.role === "user" ? (
                <div key={msg.id} className="flex justify-end">
                  <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-indigo-600 px-3.5 py-2 text-sm text-white">
                    {msg.question}
                  </p>
                </div>
              ) : msg.error ? (
                <div key={msg.id} className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {msg.error}
                </div>
              ) : msg.evidenceSufficient === false ? (
                <div key={msg.id} className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <p className="text-sm font-semibold text-amber-800">⚠ Not enough evidence</p>
                    <ModeChips mode={msg.queryMode} completeness={msg.completeness} />
                  </div>
                  <p className="mt-1 text-sm text-amber-700">{msg.reason}</p>
                  <p className="mt-2 text-xs text-amber-600">
                    Try asking about something explicitly discussed in the document.
                  </p>
                </div>
              ) : (
                <div key={msg.id} className="rounded-xl border border-slate-200 bg-white px-4 py-3">
                  <div className="mb-2 flex flex-wrap items-center gap-1.5">
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${
                        msg.cacheHit
                          ? "bg-amber-50 text-amber-700"
                          : "bg-emerald-50 text-emerald-600"
                      }`}
                    >
                      {msg.cacheHit ? "⚡ Reused previous answer" : "✓ Answered from PDF"}
                    </span>
                    <ModeChips mode={msg.queryMode} completeness={msg.completeness} />
                  </div>
                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-700">
                    {msg.answer}
                  </p>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 border-t border-slate-100 pt-2">
                      <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                        Sources
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {msg.sources.map((source) => (
                          <button
                            key={`${msg.id}-source-${source.chunk_id}`}
                            type="button"
                            onClick={() => onNavigatePage(source.page)}
                            title="Jump to source page"
                            className="rounded-full border border-indigo-200 bg-indigo-50 px-2.5 py-1 text-xs font-medium text-indigo-700 transition hover:bg-indigo-100"
                          >
                            p. {source.page}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )
            )}

            {sending && (
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-500" />
                Thinking…
              </div>
            )}
          </div>

          <form onSubmit={submit} className="border-t border-slate-100 p-3">
            <div className="flex items-end gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 focus-within:border-indigo-400">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    submit(e);
                  }
                }}
                rows={1}
                placeholder="Ask about this book…"
                className="max-h-28 min-h-8 flex-1 resize-none bg-transparent py-1 text-sm text-slate-800 outline-none placeholder:text-slate-400"
              />
              <button
                type="submit"
                disabled={sending || !input.trim()}
                aria-label="Send"
                className="rounded-lg bg-indigo-600 p-1.5 text-white transition hover:bg-indigo-500 disabled:opacity-40"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="h-4 w-4">
                  <path d="M5 12h14M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            </div>
          </form>
        </section>
      ) : (
        <button
          type="button"
          onClick={onToggle}
          aria-label="Open chat"
          className="fixed bottom-5 right-5 z-40 flex h-14 w-14 items-center justify-center rounded-full bg-indigo-600 text-2xl text-white shadow-xl transition hover:scale-105 hover:bg-indigo-500"
        >
          ✦
        </button>
      )}
    </>
  );
}
