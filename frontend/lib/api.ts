import type { ChatResult, UploadResult } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface ChatContext {
  current_page?: number;
  selected_text?: string;
}

async function parseError(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail))
      return data.detail.map((d: { msg?: string }) => d.msg ?? "").join("; ");
    return `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

export async function uploadPdf(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function askQuestion(
  documentId: string,
  question: string,
  context?: ChatContext
): Promise<ChatResult> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_id: documentId,
      question,
      ...(context?.current_page ? { current_page: context.current_page } : {}),
      ...(context?.selected_text ? { selected_text: context.selected_text } : {}),
    }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}
