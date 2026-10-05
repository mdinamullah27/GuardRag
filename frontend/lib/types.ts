export interface DocInfo {
  document_id: string;
  filename: string;
  pages: number;
}

export interface UploadResult extends DocInfo {
  status: "ready" | "already_indexed";
  reused: boolean;
  warning?: string | null;
}

export interface Source {
  page: number;
  chunk_id: number;
}

export type QueryMode =
  | "NORMAL"
  | "EXHAUSTIVE"
  | "COUNT"
  | "EXACT"
  | "PRESENCE"
  | "LOCATION"
  | "GLOBAL"
  | "CHAPTER"
  | "COMPARISON"
  | "MULTI_HOP"
  | "METADATA"
  | "REFERENCE"
  | "DEFINITION";

export type Completeness = "complete" | "partial";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  question?: string;
  answer?: string | null;
  reason?: string | null;
  cacheHit?: boolean;
  evidenceSufficient?: boolean;
  sources?: Source[];
  error?: string;
  queryMode?: QueryMode;
  completeness?: Completeness | null;
}

export interface ChatResult {
  status: "answered" | "insufficient_evidence";
  answer: string | null;
  cache_hit: boolean;
  evidence_sufficient: boolean;
  sources: Source[];
  reason: string | null;
  query_mode: QueryMode;
  completeness: Completeness | null;
}
