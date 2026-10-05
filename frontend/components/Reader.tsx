"use client";

import "pdfjs-dist/web/pdf_viewer.css";

import {
  FormEvent,
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import type { PDFDocumentProxy, PDFDocumentLoadingTask } from "pdfjs-dist";

export interface ReaderHandle {
  goToPage(page: number): void;
}

interface Props {
  file: File;
  title: string;
  pageCount: number;
  onCurrentPageChange: (page: number) => void;
  onSelection: (text: string, rect: { x: number; y: number }) => void;
  onOpenAnother: () => void;
}

type PdfjsModule = typeof import("pdfjs-dist");
type LoadState = "loading" | "ready" | "error";

const MIN_SCALE = 0.5;
const MAX_SCALE = 3;

const clamp = (value: number) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, value));

function Icon({ d, className = "h-4 w-4" }: { d: string; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={className}>
      <path d={d} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

const ICONS = {
  menu: "M4 6h16M4 12h16M4 18h16",
  close: "M6 6l12 12M18 6L6 18",
  left: "M15 18l-6-6 6-6",
  right: "M9 6l6 6-6 6",
  minus: "M5 12h14",
  plus: "M12 5v14M5 12h14",
  fit: "M4 9V5h4M20 9V5h-4M4 15v4h4M20 15v4h-4",
  fullscreen: "M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3",
  upload: "M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2M12 4v12m0-12L8 8m4-4 4 4",
};

function PageView({
  pdfjs,
  pdf,
  pageNumber,
  scale,
  root,
  registerRef,
}: {
  pdfjs: PdfjsModule;
  pdf: PDFDocumentProxy;
  pageNumber: number;
  scale: number;
  root: HTMLDivElement | null;
  registerRef: (page: number, el: HTMLDivElement | null) => void;
}) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  const renderTaskRef = useRef<{ cancel: () => void } | null>(null);
  const [inView, setInView] = useState(false);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setInView(true);
          observer.disconnect();
        }
      },
      { root, rootMargin: "600px 0px" }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [root]);

  useEffect(() => {
    if (!inView) return;
    let cancelled = false;
    (async () => {
      const page = await pdf.getPage(pageNumber);
      if (cancelled) return;
      const viewport = page.getViewport({ scale });
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = Math.floor(viewport.width * dpr);
      canvas.height = Math.floor(viewport.height * dpr);
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      renderTaskRef.current?.cancel();
      const task = page.render({
        canvasContext: ctx,
        viewport,
        transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : undefined,
      });
      renderTaskRef.current = task as unknown as { cancel: () => void };
      await task.promise;
      if (cancelled) return;
      setSize({ w: viewport.width, h: viewport.height });

      const textDiv = textRef.current;
      if (textDiv) {
        textDiv.innerHTML = "";
        textDiv.style.setProperty("--scale-factor", String(viewport.scale));
        const content = await page.getTextContent();
        if (cancelled) return;
        const layer = new pdfjs.TextLayer({
          textContentSource: content,
          container: textDiv,
          viewport,
        });
        await layer.render();
      }
    })().catch((err) => {
      if (!cancelled && err?.name !== "RenderingCancelledException") {
        console.warn("Page render failed", pageNumber, err);
      }
    });
    return () => {
      cancelled = true;
      try {
        renderTaskRef.current?.cancel();
      } catch {
        /* render task already finished */
      }
    };
  }, [pdfjs, pdf, pageNumber, scale, inView]);

  return (
    <div
      ref={(el) => {
        wrapperRef.current = el;
        registerRef(pageNumber, el);
      }}
      data-page={pageNumber}
      className="relative mx-auto shadow-sm ring-1 ring-slate-200/80"
      style={size ? { width: size.w, height: size.h } : { width: 200, height: 280 }}
    >
      <canvas ref={canvasRef} className="absolute inset-0" />
      <div ref={textRef} className="textLayer absolute inset-0" />
      {!size && (
        <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-400">
          Page {pageNumber}
        </div>
      )}
    </div>
  );
}

const Reader = forwardRef<ReaderHandle, Props>(function Reader(
  { file, title, pageCount, onCurrentPageChange, onSelection, onOpenAnother },
  ref
) {
  const rootRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  const loadingTaskRef = useRef<PDFDocumentLoadingTask | null>(null);
  const pdfjsRef = useRef<PdfjsModule | null>(null);
  const baseWidthRef = useRef(0);
  const currentRef = useRef(1);

  const [pdfjs, setPdfjs] = useState<PdfjsModule | null>(null);
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [scale, setScale] = useState(1);
  const [fitWidth, setFitWidth] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageInput, setPageInput] = useState("1");
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    if (window.innerWidth >= 768) setSidebarOpen(true);
  }, []);

  const applyFit = useCallback(() => {
    const width = scrollRef.current?.clientWidth;
    if (width && baseWidthRef.current) {
      setScale(clamp((width - 48) / baseWidthRef.current));
    }
  }, []);

  // Load the PDF once per file.
  useEffect(() => {
    let cancelled = false;
    setLoadState("loading");
    (async () => {
      const module = await import("pdfjs-dist");
      module.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
      const data = await file.arrayBuffer();
      const task = module.getDocument({ data: new Uint8Array(data) });
      const doc = await task.promise;
      if (cancelled) {
        void doc.destroy();
        return;
      }
      const first = await doc.getPage(1);
      baseWidthRef.current = first.getViewport({ scale: 1 }).width;
      pdfjsRef.current = module;
      pdfRef.current = doc;
      loadingTaskRef.current = task;
      setPdfjs(module);
      setPdf(doc);
      setLoadState("ready");
    })().catch((err) => {
      console.error("Failed to load PDF", err);
      if (!cancelled) setLoadState("error");
    });
    return () => {
      cancelled = true;
      try {
        pdfRef.current?.destroy();
      } catch {
        /* already destroyed */
      }
      pdfRef.current = null;
      setPdf(null);
    };
  }, [file]);

  // Fit-width keeps the page as wide as the available reader area.
  useEffect(() => {
    if (!fitWidth || loadState !== "ready") return;
    applyFit();
    const observer = new ResizeObserver(() => applyFit());
    if (scrollRef.current) observer.observe(scrollRef.current);
    return () => observer.disconnect();
  }, [fitWidth, loadState, applyFit]);

  const registerRef = useCallback((page: number, el: HTMLDivElement | null) => {
    if (el) pageRefs.current.set(page, el);
    else pageRefs.current.delete(page);
  }, []);

  const goToPage = useCallback(
    (page: number) => {
      const target = Math.min(pageCount, Math.max(1, page));
      const el = pageRefs.current.get(target);
      const container = scrollRef.current;
      if (el && container) {
        container.scrollTo({ top: Math.max(0, el.offsetTop - 24), behavior: "smooth" });
      }
      currentRef.current = target;
      setCurrentPage(target);
      setPageInput(String(target));
      onCurrentPageChange(target);
    },
    [pageCount, onCurrentPageChange]
  );

  useImperativeHandle(ref, () => ({ goToPage }), [goToPage]);

  const handleScroll = useCallback(() => {
    const container = scrollRef.current;
    if (!container) return;
    const marker = container.scrollTop + container.clientHeight * 0.3;
    let active = 1;
    pageRefs.current.forEach((el, page) => {
      if (el.offsetTop <= marker) active = page;
    });
    if (active !== currentRef.current) {
      currentRef.current = active;
      setCurrentPage(active);
      setPageInput(String(active));
      onCurrentPageChange(active);
    }
  }, [onCurrentPageChange]);

  const submitPage = (e: FormEvent) => {
    e.preventDefault();
    const page = parseInt(pageInput, 10);
    if (!Number.isNaN(page)) goToPage(page);
  };

  const zoomBy = (delta: number) => {
    setFitWidth(false);
    setScale((s) => clamp(Number((s + delta).toFixed(2))));
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) void rootRef.current?.requestFullscreen?.();
    else void document.exitFullscreen?.();
  };

  useEffect(() => {
    const onChange = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const handleSelection = () => {
    const selection = window.getSelection();
    const text = selection?.toString().trim() ?? "";
    if (text.length >= 3 && selection && selection.rangeCount > 0) {
      const rect = selection.getRangeAt(0).getBoundingClientRect();
      onSelection(text, { x: rect.left + rect.width / 2, y: rect.top });
    } else {
      onSelection("", { x: 0, y: 0 });
    }
  };

  const zoomLabel = `${Math.round(scale * 100)}%`;

  return (
    <div ref={rootRef} className="flex h-screen flex-col bg-slate-50 text-slate-900">
      {/* Header */}
      <header className="z-20 flex h-12 shrink-0 items-center gap-3 border-b border-slate-200 bg-white px-3 sm:px-4">
        <button
          type="button"
          onClick={() => setSidebarOpen((v) => !v)}
          title="Toggle document navigation"
          aria-label="Toggle document navigation"
          className="rounded-lg p-1.5 text-slate-500 transition hover:bg-slate-100 hover:text-slate-800"
        >
          <Icon d={ICONS.menu} className="h-5 w-5" />
        </button>
        <span className="font-semibold tracking-tight text-slate-800">
          <span className="text-indigo-600">✦</span> GuardRAG
        </span>
        <span className="hidden min-w-0 flex-1 truncate text-center text-sm text-slate-500 sm:block">
          {title}
        </span>
        <span className="min-w-0 flex-1 truncate text-sm text-slate-500 sm:hidden">{title}</span>
        <button
          type="button"
          onClick={toggleFullscreen}
          title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
          className="rounded-lg p-1.5 text-slate-500 transition hover:bg-slate-100 hover:text-slate-800"
        >
          <Icon d={isFullscreen ? ICONS.fit : ICONS.fullscreen} className="h-5 w-5" />
        </button>
        <button
          type="button"
          onClick={onOpenAnother}
          title="Open another PDF"
          className="rounded-lg p-1.5 text-slate-500 transition hover:bg-slate-100 hover:text-slate-800"
        >
          <Icon d={ICONS.upload} className="h-5 w-5" />
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* Sidebar */}
        {sidebarOpen && (
          <>
            <div
              className="fixed inset-0 z-30 bg-slate-900/30 md:hidden"
              onClick={() => setSidebarOpen(false)}
            />
            <aside className="fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-slate-200 bg-white pt-12 md:static md:z-auto md:w-56 md:shrink-0 md:pt-0">
              <div className="border-b border-slate-100 px-4 py-4">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                  Document
                </p>
                <p className="mt-1 line-clamp-2 text-sm font-medium text-slate-800">{title}</p>
                <p className="mt-1 text-xs text-slate-400">{pageCount} pages</p>
              </div>
              <div className="border-b border-slate-100 px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Pages
              </div>
              <nav className="flex-1 overflow-y-auto px-2 py-2">
                <div className="grid grid-cols-4 gap-1 md:grid-cols-3">
                  {Array.from({ length: pageCount }, (_, i) => i + 1).map((page) => (
                    <button
                      key={page}
                      type="button"
                      onClick={() => {
                        goToPage(page);
                        if (window.innerWidth < 768) setSidebarOpen(false);
                      }}
                      className={`rounded-md py-1.5 text-sm transition ${
                        currentPage === page
                          ? "bg-indigo-600 font-semibold text-white"
                          : "text-slate-600 hover:bg-slate-100"
                      }`}
                    >
                      {page}
                    </button>
                  ))}
                </div>
              </nav>
            </aside>
          </>
        )}

        {/* Reader column */}
        <div className="flex min-w-0 flex-1 flex-col">
          {/* Toolbar */}
          <div className="z-10 flex h-11 shrink-0 items-center justify-between gap-2 border-b border-slate-200 bg-white px-2 sm:px-3">
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => goToPage(currentPage - 1)}
                disabled={currentPage <= 1}
                title="Previous page"
                className="rounded-md p-1.5 text-slate-500 transition hover:bg-slate-100 disabled:opacity-30"
              >
                <Icon d={ICONS.left} />
              </button>
              <form onSubmit={submitPage} className="flex items-center gap-1 text-sm text-slate-600">
                <input
                  value={pageInput}
                  onChange={(e) => setPageInput(e.target.value.replace(/\D/g, ""))}
                  onBlur={() => setPageInput(String(currentPage))}
                  aria-label="Page number"
                  className="w-10 rounded-md border border-slate-200 px-1.5 py-1 text-center text-sm outline-none focus:border-indigo-400"
                />
                <span className="text-slate-400">/ {pageCount}</span>
              </form>
              <button
                type="button"
                onClick={() => goToPage(currentPage + 1)}
                disabled={currentPage >= pageCount}
                title="Next page"
                className="rounded-md p-1.5 text-slate-500 transition hover:bg-slate-100 disabled:opacity-30"
              >
                <Icon d={ICONS.right} />
              </button>
            </div>

            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => zoomBy(-0.1)}
                title="Zoom out"
                className="rounded-md p-1.5 text-slate-500 transition hover:bg-slate-100"
              >
                <Icon d={ICONS.minus} />
              </button>
              <span className="w-12 text-center text-sm tabular-nums text-slate-600">{zoomLabel}</span>
              <button
                type="button"
                onClick={() => zoomBy(0.1)}
                title="Zoom in"
                className="rounded-md p-1.5 text-slate-500 transition hover:bg-slate-100"
              >
                <Icon d={ICONS.plus} />
              </button>
              <button
                type="button"
                onClick={() => {
                  setFitWidth(true);
                  applyFit();
                }}
                title="Fit width"
                className={`hidden rounded-md px-2 py-1 text-sm transition sm:block ${
                  fitWidth
                    ? "bg-indigo-50 font-medium text-indigo-600"
                    : "text-slate-500 hover:bg-slate-100"
                }`}
              >
                Fit width
              </button>
              <button
                type="button"
                onClick={toggleFullscreen}
                title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
                className="rounded-md p-1.5 text-slate-500 transition hover:bg-slate-100"
              >
                <Icon d={isFullscreen ? ICONS.fit : ICONS.fullscreen} />
              </button>
            </div>
          </div>

          {/* Pages */}
          <div
            ref={scrollRef}
            onScroll={handleScroll}
            onMouseUp={handleSelection}
            className="reader-scroll relative min-h-0 flex-1 overflow-y-auto overscroll-contain bg-slate-100 px-4 py-6"
          >
            {loadState === "loading" && (
              <div className="flex h-full items-center justify-center text-sm text-slate-500">
                <span className="mr-2 h-2 w-2 animate-pulse rounded-full bg-indigo-500" />
                Loading document…
              </div>
            )}
            {loadState === "error" && (
              <div className="flex h-full items-center justify-center text-sm text-red-600">
                Could not open this PDF in the reader.
              </div>
            )}
            {loadState === "ready" && pdf && pdfjs && (
              <div className="mx-auto flex w-fit flex-col items-center gap-5">
                {Array.from({ length: pageCount }, (_, i) => i + 1).map((page) => (
                  <PageView
                    key={`${page}-${pdf.fingerprints?.[0] ?? "doc"}`}
                    pdfjs={pdfjs}
                    pdf={pdf}
                    pageNumber={page}
                    scale={scale}
                    root={scrollRef.current}
                    registerRef={registerRef}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
});

export default Reader;
