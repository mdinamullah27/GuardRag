"use client";

import FloatingChat from "@/components/FloatingChat";
import Reader, { type ReaderHandle } from "@/components/Reader";
import UploadScreen from "@/components/UploadScreen";
import { askQuestion, uploadPdf } from "@/lib/api";
import type { ChatMessage, DocInfo } from "@/lib/types";
import { useEffect, useRef, useState } from "react";

let idCounter = 0;
const nextId = () => String(++idCounter);

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [doc, setDoc] = useState<DocInfo | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [toast, setToast] = useState<{ kind: "success" | "warning"; text: string } | null>(null);

  const [chatOpen, setChatOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sending, setSending] = useState(false);

  const [selectedText, setSelectedText] = useState<string | null>(null);
  const [selectionMenu, setSelectionMenu] = useState<{ x: number; y: number } | null>(null);

  const readerRef = useRef<ReaderHandle>(null);
  const currentPageRef = useRef(1);
  const selectedTextRef = useRef<string | null>(null);

  useEffect(() => {
    selectedTextRef.current = selectedText;
  }, [selectedText]);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), toast.kind === "warning" ? 9000 : 4000);
    return () => clearTimeout(timer);
  }, [toast]);

  const handleUpload = async (uploadFile: File) => {
    setUploading(true);
    setUploadError(null);
    try {
      const result = await uploadPdf(uploadFile);
      setFile(uploadFile);
      setDoc(result);
      currentPageRef.current = 1;
      setMessages([]);
      setChatOpen(false);
      setSelectedText(null);
      setSelectionMenu(null);
      if (result.warning) {
        setToast({ kind: "warning", text: result.warning });
      } else if (result.reused) {
        setToast({
          kind: "success",
          text: "Already indexed — reusing existing embeddings, no re-chunking.",
        });
      }
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setDoc(null);
    setMessages([]);
    setChatOpen(false);
    setSelectedText(null);
    setSelectionMenu(null);
    setUploadError(null);
  };

  const handleSend = async (question: string) => {
    if (!doc) return;
    const contextSelection = selectedTextRef.current;
    const placeholderId = nextId();
    setMessages((prev) => [
      ...prev,
      { id: nextId(), role: "user", question },
      { id: placeholderId, role: "assistant" },
    ]);
    setSending(true);
    setSelectedText(null);
    setSelectionMenu(null);
    try {
      const result = await askQuestion(doc.document_id, question, {
        current_page: currentPageRef.current,
        selected_text: contextSelection ?? undefined,
      });
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === placeholderId
            ? {
                ...msg,
                answer: result.answer,
                reason: result.reason,
                cacheHit: result.cache_hit,
                evidenceSufficient: result.evidence_sufficient,
                sources: result.sources,
                queryMode: result.query_mode,
                completeness: result.completeness,
              }
            : msg
        )
      );
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Request failed";
      setMessages((prev) =>
        prev.map((msg) => (msg.id === placeholderId ? { ...msg, error: detail } : msg))
      );
    } finally {
      setSending(false);
    }
  };

  const handleSelection = (text: string, rect: { x: number; y: number }) => {
    if (text) setSelectionMenu(rect);
    else setSelectionMenu(null);
  };

  const confirmSelection = () => {
    const text = window.getSelection()?.toString().trim() || null;
    if (text) {
      setSelectedText(text);
      setChatOpen(true);
    }
    setSelectionMenu(null);
  };

  if (!doc || !file) {
    return <UploadScreen uploading={uploading} error={uploadError} onFile={handleUpload} />;
  }

  return (
    <>
      <Reader
        ref={readerRef}
        file={file}
        title={doc.filename}
        pageCount={doc.pages}
        onCurrentPageChange={(page) => {
          currentPageRef.current = page;
        }}
        onSelection={handleSelection}
        onOpenAnother={reset}
      />

      {selectionMenu && (
        <button
          type="button"
          onMouseDown={(e) => e.preventDefault()}
          onClick={confirmSelection}
          className="fixed z-50 -translate-x-1/2 -translate-y-full rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-medium text-white shadow-lg transition hover:bg-indigo-600"
          style={{ left: selectionMenu.x, top: selectionMenu.y - 8 }}
        >
          ✨ Ask GuardRAG
        </button>
      )}

      <FloatingChat
        open={chatOpen}
        onToggle={() => setChatOpen((v) => !v)}
        messages={messages}
        sending={sending}
        selectedText={selectedText}
        onSend={handleSend}
        onNavigatePage={(page) => readerRef.current?.goToPage(page)}
        onDismissSelection={() => setSelectedText(null)}
      />

      {toast && (
        <div
          className={`fixed left-1/2 top-4 z-50 max-w-[min(32rem,calc(100vw-2rem))] -translate-x-1/2 rounded-xl border bg-white px-4 py-2.5 text-sm font-medium shadow-lg ${
            toast.kind === "warning" ? "border-amber-300" : "border-emerald-200"
          }`}
        >
          <span className={`mr-1.5 ${toast.kind === "warning" ? "text-amber-500" : "text-emerald-500"}`}>
            {toast.kind === "warning" ? "⚠" : "✓"}
          </span>
          <span className="text-slate-700">{toast.text}</span>
        </div>
      )}
    </>
  );
}
