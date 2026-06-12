import React, { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, asPageResult, pageCount } from "./api";
import {
  encodePath,
  formatSessionKeyForTable,
  proactiveFlowLabel,
  proactiveResultLabel,
  proactiveSectionLabel,
  proactiveTickPreview,
  relativeTime,
  renderMarkdown,
  roleClass,
  shortTs,
  stripMarkdown,
} from "./format";
import { attachJsonViewers, installDashboardGlobals, jvPlaceholder, loadPluginAssets } from "./pluginRuntime";
import { PluginDetail } from "./PluginDetail";
import type {
  DashboardColumn,
  DailyWorkspaceSnapshot,
  KnowledgeChunkRow,
  KnowledgeDocumentRow,
  KnowledgeRetrievalEventRow,
  MessageRow,
  PageResult,
  PluginBatchAction,
  PluginConfig,
  PluginDispatch,
  PluginState,
  ProactiveOverview,
  ProactiveStep,
  ProactiveTick,
  SessionRow,
  SortOrder,
  ViewMode,
} from "./types";

type NavOpen = Record<string, boolean>;

// Creates a PluginDispatch bound to the given plugin + latest state getter.
function makeDispatch(
  plugin: PluginConfig,
  getState: () => PluginState | null,
  onSetState: (updater: (s: PluginState) => PluginState) => void,
  onActivate?: () => void,
): PluginDispatch {
  const fetchAndApply = async (
    nextFilters: Record<string, string>,
    nextSortBy: string,
    nextSortOrder: SortOrder,
  ): Promise<void> => {
    const state = getState();
    if (!state) return;
    const result = await plugin.fetchPage({ page: 1, pageSize: state.pageSize, filters: nextFilters, sortBy: nextSortBy, sortOrder: nextSortOrder });
    onSetState((s) => ({
      ...s,
      page: 1,
      total: result.total || 0,
      items: result.items || [],
      activeRowKey: null,
      activeDetail: null,
      filters: nextFilters,
      sortBy: nextSortBy,
      sortOrder: nextSortOrder,
    }));
  };

  const updateFilters = (updater: (filters: Record<string, string>) => Record<string, string>): void => {
    const state = getState();
    if (!state) return;
    void fetchAndApply(updater({ ...state.filters }), state.sortBy, state.sortOrder);
  };

  return {
    get filters() { return getState()?.filters ?? {}; },
    setFilter(key: string, value: string): void {
      updateFilters((filters) => ({ ...filters, [key]: value }));
    },
    clearFilter(key: string): void {
      updateFilters((filters) => {
        delete filters[key];
        return filters;
      });
    },
    setFilters(next: Record<string, string>): void {
      updateFilters((filters) => ({ ...filters, ...next }));
    },
    clearFilters(keys: string[]): void {
      updateFilters((filters) => {
        for (const key of keys) delete filters[key];
        return filters;
      });
    },
    get sortBy() { return getState()?.sortBy ?? ""; },
    get sortOrder() { return getState()?.sortOrder ?? "desc"; },
    setSort(key: string): void {
      const state = getState();
      if (!state) return;
      const nextOrder: SortOrder = state.sortBy === key && state.sortOrder === "desc" ? "asc" : "desc";
      void fetchAndApply(state.filters, key, nextOrder);
    },
    refresh(): void {
      const state = getState();
      if (!state) return;
      void fetchAndApply(state.filters, state.sortBy, state.sortOrder);
    },
    activate(): void {
      onActivate?.();
    },
  };
}

function App(): React.ReactElement {
  const isDailyShowcase = window.location.pathname === "/daily/showcase" || window.location.pathname === "/workspace/showcase";
  const initialView: ViewMode = window.location.pathname === "/daily" || window.location.pathname === "/workspace" || isDailyShowcase ? "daily" : "sessions";
  const [viewMode, setViewMode] = useState<ViewMode>(initialView);
  const [navOpen, setNavOpen] = useState<NavOpen>({ sessions: false, proactive: false, daily: true });
  const [plugins, setPlugins] = useState<PluginConfig[]>([]);
  const [pluginState, setPluginState] = useState<Record<string, PluginState>>({});
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [sessionSearch, setSessionSearch] = useState("");
  const [sessionChannel, setSessionChannel] = useState("");
  const [activeSessionKey, setActiveSessionKey] = useState<string | null>(null);
  const [activeSession, setActiveSession] = useState<SessionRow | null>(null);
  const [messages, setMessages] = useState<MessageRow[]>([]);
  const [messageSearch, setMessageSearch] = useState("");
  const [messageRole, setMessageRole] = useState("");
  const [messagePage, setMessagePage] = useState(1);
  const [messageSortBy, setMessageSortBy] = useState("ts");
  const [messageSortOrder, setMessageSortOrder] = useState<SortOrder>("desc");
  const [totalMessages, setTotalMessages] = useState(0);
  const [activeMessage, setActiveMessage] = useState<MessageRow | null>(null);
  const [selectedMessageIds, setSelectedMessageIds] = useState<Set<string>>(new Set());
  const [proactiveOverview, setProactiveOverview] = useState<ProactiveOverview | null>(null);
  const [proactiveSection, setProactiveSection] = useState("all");
  const [proactiveItems, setProactiveItems] = useState<ProactiveTick[]>([]);
  const [proactivePage, setProactivePage] = useState(1);
  const [proactiveSortBy, setProactiveSortBy] = useState("started_at");
  const [proactiveSortOrder, setProactiveSortOrder] = useState<SortOrder>("desc");
  const [proactiveTotal, setProactiveTotal] = useState(0);
  const [proactiveSessionFilter, setProactiveSessionFilter] = useState("");
  const [knowledgeDocuments, setKnowledgeDocuments] = useState<KnowledgeDocumentRow[]>([]);
  const [knowledgeChunks, setKnowledgeChunks] = useState<KnowledgeChunkRow[]>([]);
  const [knowledgeEvents, setKnowledgeEvents] = useState<KnowledgeRetrievalEventRow[]>([]);
  const [knowledgeVectorBackend, setKnowledgeVectorBackend] = useState("");
  const [dailySnapshot, setDailySnapshot] = useState<DailyWorkspaceSnapshot | null>(null);
  const [dailyDate, setDailyDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [activeKnowledgeDocumentId, setActiveKnowledgeDocumentId] = useState<string>("");
  const [activeProactiveKey, setActiveProactiveKey] = useState<string | null>(null);
  const [activeProactiveDetail, setActiveProactiveDetail] = useState<ProactiveTick | null>(null);
  const [activeProactiveSteps, setActiveProactiveSteps] = useState<ProactiveStep[]>([]);
  const [hiddenPlugins, setHiddenPlugins] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);

  const messagePageSize = 25;
  const proactivePageSize = 25;
  const currentPluginId = viewMode.startsWith("plugin:") ? viewMode.slice(7) : "";
  const currentPlugin = plugins.find((plugin) => plugin.id === currentPluginId) ?? null;
  const currentPluginState = currentPluginId ? pluginState[currentPluginId] : null;

  const channels = useMemo(() => Array.from(new Set(sessions.map((session) => session.key.split(":")[0]).filter(Boolean))), [sessions]);

  const run = useCallback(async (work: () => Promise<void>) => {
    try {
      setError(null);
      await work();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }, []);

  const loadSessions = useCallback(async () => {
    const params = new URLSearchParams();
    if (sessionSearch) params.set("q", sessionSearch);
    if (sessionChannel) params.set("channel", sessionChannel);
    params.set("page_size", "200");
    const payload = asPageResult(await api<PageResult<SessionRow>>(`/api/dashboard/sessions?${params.toString()}`));
    setSessions(payload.items);
    setActiveSession((current) => {
      if (!activeSessionKey) return current;
      return payload.items.find((session) => session.key === activeSessionKey) ?? null;
    });
  }, [activeSessionKey, sessionChannel, sessionSearch]);

  const loadMessages = useCallback(async () => {
    const params = new URLSearchParams();
    if (activeSessionKey) params.set("session_key", activeSessionKey);
    if (messageSearch) params.set("q", messageSearch);
    if (messageRole) params.set("role", messageRole);
    params.set("page", String(messagePage));
    params.set("page_size", String(messagePageSize));
    params.set("sort_by", messageSortBy);
    params.set("sort_order", messageSortOrder);
    const payload = asPageResult(await api<PageResult<MessageRow>>(`/api/dashboard/messages?${params.toString()}`));
    setMessages(payload.items);
    setTotalMessages(payload.total);
    setActiveMessage((current) => current && payload.items.some((item) => item.id === current.id) ? current : null);
  }, [activeSessionKey, messagePage, messageRole, messageSearch, messageSortBy, messageSortOrder]);

  const loadProactiveOverview = useCallback(async () => {
    setProactiveOverview(await api<ProactiveOverview>("/api/dashboard/proactive/overview"));
  }, []);

  const loadProactivePanel = useCallback(async () => {
    const params = new URLSearchParams();
    params.set("page", String(proactivePage));
    params.set("page_size", String(proactivePageSize));
    params.set("sort_by", proactiveSortBy);
    params.set("sort_order", proactiveSortOrder);
    if (proactiveSessionFilter) params.set("session_key", proactiveSessionFilter);
    if (proactiveSection === "reply" || proactiveSection === "skip") params.set("terminal_action", proactiveSection);
    if (proactiveSection === "drift" || proactiveSection === "proactive") params.set("flow", proactiveSection);
    if (["busy", "cooldown", "presence"].includes(proactiveSection)) params.set("gate_exit", proactiveSection);
    const payload = asPageResult(await api<PageResult<ProactiveTick>>(`/api/dashboard/proactive/tick_logs?${params.toString()}`));
    setProactiveItems(payload.items);
    setProactiveTotal(payload.total);
    setActiveProactiveKey((current) => current && payload.items.some((item) => item.tick_id === current) ? current : null);
  }, [proactivePage, proactiveSection, proactiveSessionFilter, proactiveSortBy, proactiveSortOrder]);

  const loadKnowledgePanel = useCallback(async () => {
    const docsPayload = await api<PageResult<KnowledgeDocumentRow> & { vector_backend?: string }>("/api/dashboard/knowledge/documents");
    const documents = docsPayload.items ?? [];
    setKnowledgeDocuments(documents);
    setKnowledgeVectorBackend(docsPayload.vector_backend ?? "");
    const selectedDocument = activeKnowledgeDocumentId && documents.some((item) => item.id === activeKnowledgeDocumentId)
      ? activeKnowledgeDocumentId
      : documents[0]?.id ?? "";
    setActiveKnowledgeDocumentId(selectedDocument);
    const chunkParams = new URLSearchParams();
    if (selectedDocument) chunkParams.set("document_id", selectedDocument);
    chunkParams.set("limit", "100");
    const [chunksPayload, eventsPayload] = await Promise.all([
      api<PageResult<KnowledgeChunkRow>>(`/api/dashboard/knowledge/chunks?${chunkParams.toString()}`),
      api<PageResult<KnowledgeRetrievalEventRow>>("/api/dashboard/knowledge/retrieval-events?limit=50"),
    ]);
    setKnowledgeChunks(chunksPayload.items ?? []);
    setKnowledgeEvents(eventsPayload.items ?? []);
  }, [activeKnowledgeDocumentId]);

  const loadDailyWorkspace = useCallback(async (dateValue = dailyDate) => {
    const params = new URLSearchParams();
    if (dateValue) params.set("date", dateValue);
    const snapshot = await api<DailyWorkspaceSnapshot>(`/api/dashboard/daily-workspace?${params.toString()}`);
    setDailySnapshot(snapshot);
    setDailyDate(snapshot.date);
  }, [dailyDate]);

  const loadPluginPanel = useCallback(async (pluginId: string) => {
    const plugin = plugins.find((item) => item.id === pluginId);
    const state = pluginState[pluginId];
    if (!plugin || !state) return;
    const result = await plugin.fetchPage({ page: state.page, pageSize: state.pageSize, filters: state.filters, sortBy: state.sortBy, sortOrder: state.sortOrder });
    setPluginState((current) => ({
      ...current,
      [pluginId]: {
        ...current[pluginId],
        total: result.total || 0,
        items: result.items || [],
        activeRowKey: current[pluginId]?.activeRowKey && result.items.some((item) => String(item[plugin.rowKey] ?? "") === current[pluginId].activeRowKey)
          ? current[pluginId].activeRowKey
          : null,
        activeDetail: current[pluginId]?.activeRowKey && result.items.some((item) => String(item[plugin.rowKey] ?? "") === current[pluginId].activeRowKey)
          ? current[pluginId].activeDetail
          : null,
      },
    }));
  }, [pluginState, plugins]);

  const refreshCurrentView = useCallback(async () => {
    await loadSessions();
    if (viewMode === "proactive") {
      await loadProactiveOverview();
      await loadProactivePanel();
    } else if (viewMode === "knowledge") {
      await loadKnowledgePanel();
    } else if (viewMode === "daily") {
      await loadDailyWorkspace();
    } else if (viewMode.startsWith("plugin:")) {
      await loadPluginPanel(viewMode.slice(7));
    } else {
      await loadMessages();
    }
  }, [loadDailyWorkspace, loadKnowledgePanel, loadMessages, loadPluginPanel, loadProactiveOverview, loadProactivePanel, loadSessions, viewMode]);

  useEffect(() => {
    const refresh = (): void => {
      void run(refreshCurrentView);
    };
    window.addEventListener("kotarou-dashboard-refresh", refresh);
    return () => window.removeEventListener("kotarou-dashboard-refresh", refresh);
  }, [refreshCurrentView, run]);

  useEffect(() => {
    installDashboardGlobals((plugin) => {
      setPlugins((current) => current.some((item) => item.id === plugin.id) ? current : [...current, plugin]);
      setPluginState((current) => current[plugin.id] ? current : {
        ...current,
        [plugin.id]: {
          page: 1,
          pageSize: plugin.pageSize || 25,
          total: 0,
          items: [],
          activeRowKey: null,
          activeDetail: null,
          filters: {},
          sortBy: plugin.defaultSortBy ?? "",
          sortOrder: plugin.defaultSortOrder ?? "desc",
          selectedIds: new Set(),
        },
      });
    });
    void loadPluginAssets();
  }, []);

  useEffect(() => {
    void run(async () => {
      await loadSessions();
      await loadMessages();
      await loadProactiveOverview();
      if (initialView === "daily") await loadDailyWorkspace();
    });
  }, [initialView, loadDailyWorkspace, loadMessages, loadProactiveOverview, loadSessions, run]);

  useEffect(() => {
    for (const plugin of plugins) {
      void run(async () => {
        const count = await plugin.getCount();
        if (count === null) {
          setHiddenPlugins((current) => ({ ...current, [plugin.id]: true }));
        } else {
          setHiddenPlugins((current) => ({ ...current, [plugin.id]: false }));
          setPluginState((current) => ({
            ...current,
            [plugin.id]: { ...current[plugin.id], total: count },
          }));
        }
      });
    }
  }, [plugins, run]);

  const focusView = useCallback((next: ViewMode): void => {
    setViewMode(next);
    setNavOpen((current) => ({ ...current, [next]: true }));
  }, []);

  const selectView = (next: ViewMode): void => {
    focusView(next);
    void run(async () => {
      if (next === "sessions") await loadMessages();
      else if (next === "proactive") {
        await loadProactiveOverview();
        await loadProactivePanel();
      } else if (next === "knowledge") await loadKnowledgePanel();
      else if (next === "daily") await loadDailyWorkspace();
      else await loadPluginPanel(next.slice(7));
    });
  };

  const toggleNav = (kind: ViewMode): void => {
    if (viewMode !== kind) {
      selectView(kind);
      return;
    }
    setNavOpen((current) => ({ ...current, [kind]: !current[kind] }));
  };

  const sort = (scope: "messages" | "proactive", key: string): void => {
    const flip = (currentKey: string, currentOrder: SortOrder): SortOrder => currentKey === key && currentOrder === "desc" ? "asc" : "desc";
    if (scope === "messages") {
      setMessageSortOrder(flip(messageSortBy, messageSortOrder));
      setMessageSortBy(key);
      setMessagePage(1);
    } else {
      setProactiveSortOrder(flip(proactiveSortBy, proactiveSortOrder));
      setProactiveSortBy(key);
      setProactivePage(1);
    }
  };

  useEffect(() => {
    if (viewMode === "sessions") void run(loadMessages);
  }, [loadMessages, run, viewMode]);

  useEffect(() => {
    if (viewMode === "proactive") void run(loadProactivePanel);
  }, [loadProactivePanel, run, viewMode]);

  useEffect(() => {
    if (viewMode === "knowledge") void run(loadKnowledgePanel);
  }, [loadKnowledgePanel, run, viewMode]);

  useEffect(() => {
    if (viewMode === "daily") void run(() => loadDailyWorkspace(dailyDate));
  }, [dailyDate, loadDailyWorkspace, run, viewMode]);

  const currentPageCount = currentPluginState
    ? pageCount(currentPluginState.total, currentPluginState.pageSize)
    : viewMode === "daily"
      ? 1
    : viewMode === "knowledge"
      ? 1
    : viewMode === "proactive"
      ? pageCount(proactiveTotal, proactivePageSize)
      : pageCount(totalMessages, messagePageSize);

  const currentPage = currentPluginState?.page ?? (viewMode === "daily" || viewMode === "knowledge" ? 1 : viewMode === "proactive" ? proactivePage : messagePage);

  const changePage = (delta: number): void => {
    if (currentPage + delta < 1 || currentPage + delta > currentPageCount) return;
    if (currentPluginId) {
      void run(async () => {
        const plugin = plugins.find((item) => item.id === currentPluginId);
        const state = pluginState[currentPluginId];
        if (!plugin || !state) return;
        const nextPage = state.page + delta;
        const result = await plugin.fetchPage({ page: nextPage, pageSize: state.pageSize, filters: state.filters, sortBy: state.sortBy, sortOrder: state.sortOrder });
        setPluginState((current) => ({
          ...current,
          [currentPluginId]: {
            ...current[currentPluginId],
            page: nextPage,
            total: result.total || 0,
            items: result.items || [],
            activeRowKey: null,
            activeDetail: null,
          },
        }));
      });
    } else if (viewMode === "proactive") setProactivePage((page) => page + delta);
    else setMessagePage((page) => page + delta);
  };

  // Batch count: messages or plugin selectedIds
  const pluginBatchCount = currentPluginState?.selectedIds.size ?? 0;
  const batchCount = viewMode.startsWith("plugin:") ? pluginBatchCount : selectedMessageIds.size;

  // dispatch for current plugin (used in DetailPane and batch bar)
  const currentDispatch = currentPlugin && currentPluginState
    ? makeDispatch(
        currentPlugin,
        () => pluginState[currentPlugin.id] ?? null,
        (updater) => setPluginState((c) => ({ ...c, [currentPlugin.id]: updater(c[currentPlugin.id]) })),
        () => focusView(`plugin:${currentPlugin.id}`),
      )
    : undefined;

  if (isDailyShowcase) {
    return (
      <>
        <DailyShowcasePage snapshot={dailySnapshot} selectedDate={dailyDate} onSelectDate={setDailyDate} />
        {error && <div className="modal-backdrop" onClick={() => setError(null)}><div className="modal"><div className="modal-title">请求失败</div><p>{error}</p><div className="modal-actions"><button className="primary" type="button" onClick={() => setError(null)}>关闭</button></div></div></div>}
      </>
    );
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">K</div>
          <div>
            <div className="brand-title">Kotarou Dashboard</div>
            <div className="brand-sub">Session / Memory Explorer</div>
          </div>
        </div>
        <TopbarFilters
          viewMode={viewMode}
          messageSearch={messageSearch}
          setMessageSearch={(value) => { setMessageSearch(value); setMessagePage(1); }}
          messageRole={messageRole}
          setMessageRole={(value) => { setMessageRole(value); setMessagePage(1); }}
          activeSessionKey={activeSessionKey}
          clearSession={() => { setActiveSessionKey(null); setActiveSession(null); setActiveMessage(null); setMessagePage(1); }}
          proactiveSection={proactiveSection}
          proactiveSessionFilter={proactiveSessionFilter}
          clearProactiveSession={() => { setProactiveSessionFilter(""); setProactivePage(1); }}
          currentPlugin={currentPlugin}
          currentPluginState={currentPluginState}
          onSetPluginState={currentPlugin ? (updater) => setPluginState((c) => ({ ...c, [currentPlugin.id]: updater(c[currentPlugin.id]) })) : undefined}
          knowledgeVectorBackend={knowledgeVectorBackend}
          dailySnapshot={dailySnapshot}
        />
        <div className="topbar-view">
          <div className="view-chip"><span>{viewLabel(viewMode, currentPlugin)}</span></div>
          {viewMode.startsWith("plugin:") && currentPlugin?.renderTopbarAction && currentPluginState && currentDispatch && (
            <PluginTopbarAction
              plugin={currentPlugin}
              pluginId={currentPlugin.id}
              state={currentPluginState}
              onSetState={(updater) => setPluginState((c) => ({ ...c, [currentPlugin.id]: updater(c[currentPlugin.id]) }))}
              onActivate={() => focusView(`plugin:${currentPlugin.id}`)}
            />
          )}
        </div>
      </header>

      <main className={`workspace ${viewMode === "daily" ? "daily-mode" : ""}`}>
        <aside className="sessions-pane">
          <div className="pane-head">
            <div className="pane-kicker">Explorer</div>
            <div className="pane-title">
              {currentPlugin && currentPluginState
                ? (currentPlugin.countTitle ? currentPlugin.countTitle(currentPluginState.total) : `${currentPluginState.total} 条记录`)
                : `${sessions.length} 个会话`}
            </div>
          </div>
          <div className="filters-stack">
            <label className="search search-small">
              <span>⌕</span>
              <input type="text" placeholder="过滤 session" value={sessionSearch} onChange={(event) => setSessionSearch(event.target.value.trim())} />
            </label>
            <select value={sessionChannel} onChange={(event) => setSessionChannel(event.target.value)}>
              <option value="">全部 channel</option>
              {channels.map((channel) => <option key={channel} value={channel}>{channel}</option>)}
            </select>
          </div>
          <nav className="explorer-nav">
            <NavGroup label="Sessions" count={totalMessages || totalSessionMessages(sessions)} active={viewMode === "sessions"} open={!!navOpen.sessions} onToggle={() => toggleNav("sessions")}>
              <button className={`all-messages-row ${viewMode === "sessions" && !activeSessionKey ? "active" : ""}`} type="button" onClick={() => {
                setActiveSessionKey(null);
                setActiveSession(null);
                setActiveMessage(null);
                setMessagePage(1);
                selectView("sessions");
              }}>
                <span>全部消息</span><strong>{sessions.length}</strong>
              </button>
              <div className="session-list">
                {sessions.map((session) => (
                  <button key={session.key} className={`session-item ${activeSessionKey === session.key ? "active" : ""}`} type="button" onClick={() => {
                    setActiveSessionKey(session.key);
                    setActiveSession(session);
                    setActiveMessage(null);
                    setMessagePage(1);
                    selectView("sessions");
                  }}>
                    <div className="nav-item-row">
                      <span className="nav-type-dot memory-type-profile" />
                      <span className="nav-item-name mono">{formatSessionKeyForTable(session.key)}</span>
                      <span className="nav-item-count">{session.message_count}</span>
                    </div>
                    <div className="nav-item-desc">{relativeTime(session.updated_at)}</div>
                  </button>
                ))}
              </div>
            </NavGroup>
            <NavGroup label="Daily Workspace" count={dailySnapshot?.missions.length ?? 0} active={viewMode === "daily"} open={!!navOpen.daily} onToggle={() => toggleNav("daily")}>
              <button className={`all-messages-row ${viewMode === "daily" ? "active" : ""}`} type="button" onClick={() => selectView("daily")}>
                <span>今日工作台</span><strong>{dailySnapshot?.status ?? "idle"}</strong>
              </button>
              <div className="proactive-quick-list">
                {(dailySnapshot?.archive_dates ?? []).map((item) => (
                  <button key={item.date} className={`proactive-quick-item ${dailyDate === item.date ? "active" : ""}`} type="button" onClick={() => {
                    setDailyDate(item.date);
                    selectView("daily");
                  }}>
                    <div className="nav-item-row">
                      <span className="nav-item-name">{item.label}</span>
                      <span className="nav-item-count">{item.has_real_data ? item.count : "sample"}</span>
                    </div>
                  </button>
                ))}
              </div>
            </NavGroup>
            <NavGroup label="Proactive" count={proactiveOverview?.counts.tick_logs ?? proactiveTotal} active={viewMode === "proactive"} open={!!navOpen.proactive} onToggle={() => toggleNav("proactive")}>
              <button className={`all-messages-row ${proactiveSection === "all" && viewMode === "proactive" ? "active" : ""}`} type="button" onClick={() => { setProactiveSection("all"); setProactivePage(1); selectView("proactive"); }}>
                <span>{proactiveSectionLabel("all")}</span><strong>{proactiveSectionCount("all", proactiveOverview)}</strong>
              </button>
              <div className="proactive-quick-list">
                {["drift", "proactive", "reply", "skip", "busy", "cooldown", "presence"].map((section) => (
                  <button key={section} className={`proactive-quick-item ${proactiveSection === section ? "active" : ""}`} type="button" onClick={() => {
                    setProactiveSection(section);
                    setProactivePage(1);
                    selectView("proactive");
                  }}>
                    <div className="nav-item-row">
                      <span className="nav-item-name">{proactiveSectionLabel(section)}</span>
                      <span className="nav-item-count">{proactiveSectionCount(section, proactiveOverview)}</span>
                    </div>
                  </button>
                ))}
              </div>
            </NavGroup>
            <NavGroup label="Knowledge" count={knowledgeDocuments.length} active={viewMode === "knowledge"} open={!!navOpen.knowledge} onToggle={() => toggleNav("knowledge")}>
              <button className={`all-messages-row ${viewMode === "knowledge" ? "active" : ""}`} type="button" onClick={() => selectView("knowledge")}>
                <span>文档知识库</span><strong>{knowledgeDocuments.length}</strong>
              </button>
              <div className="session-list">
                {knowledgeDocuments.slice(0, 50).map((doc) => (
                  <button key={doc.id} className={`session-item ${activeKnowledgeDocumentId === doc.id ? "active" : ""}`} type="button" onClick={() => {
                    setActiveKnowledgeDocumentId(doc.id);
                    selectView("knowledge");
                  }}>
                    <div className="nav-item-row">
                      <span className="nav-type-dot memory-type-event" />
                      <span className="nav-item-name">{doc.title || doc.source_path.split("/").pop()}</span>
                      <span className="nav-item-count">{doc.chunk_count}</span>
                    </div>
                    <div className="nav-item-desc">{doc.file_type} · {relativeTime(doc.updated_at)}</div>
                  </button>
                ))}
              </div>
            </NavGroup>
            {plugins.some((p) => !hiddenPlugins[p.id]) && (
              <div className="nav-section-divider">
                <span>Plugins</span>
              </div>
            )}
            {plugins.filter((p) => !hiddenPlugins[p.id]).map((plugin) => {
              const pState = pluginState[plugin.id];
              const pDispatch = pState
                ? makeDispatch(
                    plugin,
                    () => pluginState[plugin.id] ?? null,
                    (updater) => setPluginState((c) => ({ ...c, [plugin.id]: updater(c[plugin.id]) })),
                    () => selectView(`plugin:${plugin.id}`),
                  )
                : undefined;
              const isActive = viewMode === `plugin:${plugin.id}`;
              return (
                <NavGroup key={plugin.id} label={plugin.label} count={pState?.total ?? 0} active={isActive} open={!!navOpen[`plugin:${plugin.id}`]} onToggle={() => toggleNav(`plugin:${plugin.id}`)}>
                  {plugin.renderNavBody && pState && pDispatch
                    ? <PluginNavBody
                        plugin={plugin}
                        pluginId={plugin.id}
                        state={pState}
                        onSetState={(updater) => setPluginState((c) => ({ ...c, [plugin.id]: updater(c[plugin.id]) }))}
                        onActivate={() => focusView(`plugin:${plugin.id}`)}
                      />
                    : <button className={`all-messages-row ${isActive ? "active" : ""}`} type="button" onClick={() => selectView(`plugin:${plugin.id}`)}>
                        <span>{plugin.label}</span><strong>{pState?.total ?? 0}</strong>
                      </button>
                  }
                </NavGroup>
              );
            })}
          </nav>
        </aside>

        <section className="messages-pane">
          {batchCount > 0 && (
            <div className="batch-bar">
              <span>已选 {batchCount} 条</span>
              {viewMode.startsWith("plugin:") && currentPlugin?.batchActions && currentPluginState
                ? currentPlugin.batchActions.map((action: PluginBatchAction) => (
                    <button key={action.label} className={action.className} type="button" onClick={() => void run(async () => {
                      const ids = [...currentPluginState.selectedIds];
                      await action.run(ids);
                      setPluginState((c) => ({ ...c, [currentPlugin.id]: { ...c[currentPlugin.id], selectedIds: new Set() } }));
                      await loadPluginPanel(currentPlugin.id);
                    })}>{action.label}</button>
                  ))
                : <button className="danger-ghost" type="button" onClick={() => void run(async () => {
                    await api("/api/dashboard/messages/batch-delete", { method: "POST", body: JSON.stringify({ ids: [...selectedMessageIds] }) });
                    setSelectedMessageIds(new Set());
                    await refreshCurrentView();
                  })}>批量删除</button>
              }
              <button className="ghost" type="button" onClick={() => {
                if (viewMode.startsWith("plugin:") && currentPlugin) {
                  setPluginState((c) => ({ ...c, [currentPlugin.id]: { ...c[currentPlugin.id], selectedIds: new Set() } }));
                } else {
                  setSelectedMessageIds(new Set());
                }
              }}>取消选择</button>
            </div>
          )}
          {viewMode === "knowledge" ? (
            <KnowledgePanel
              documents={knowledgeDocuments}
              chunks={knowledgeChunks}
              events={knowledgeEvents}
              vectorBackend={knowledgeVectorBackend}
              activeDocumentId={activeKnowledgeDocumentId}
              onSelectDocument={(id) => setActiveKnowledgeDocumentId(id)}
            />
          ) : viewMode === "daily" ? (
            <DailyWorkspacePanel snapshot={dailySnapshot} selectedDate={dailyDate} onSelectDate={setDailyDate} />
          ) : <>
            <TableHead viewMode={viewMode} plugin={currentPlugin} pluginState={currentPluginState} messageSortBy={messageSortBy} messageSortOrder={messageSortOrder} proactiveSortBy={proactiveSortBy} proactiveSortOrder={proactiveSortOrder} onSort={sort} onPluginSort={currentDispatch ? (key) => currentDispatch.setSort(key) : undefined} />
            <div className="table-body">
              <Rows
              viewMode={viewMode}
              messages={messages}
              proactiveItems={proactiveItems}
              plugin={currentPlugin}
              pluginState={currentPluginState}
              selectedMessageIds={selectedMessageIds}
              activeMessage={activeMessage}
              activeProactiveKey={activeProactiveKey}
              onSelectMessage={setActiveMessage}
              onSelectProactive={(item) => void run(async () => {
                setActiveProactiveKey(item.tick_id);
                const [detail, steps] = await Promise.all([
                  api<ProactiveTick>(`/api/dashboard/proactive/tick_logs/${encodePath(item.tick_id)}`),
                  api<PageResult<ProactiveStep>>(`/api/dashboard/proactive/tick_logs/${encodePath(item.tick_id)}/steps`),
                ]);
                setActiveProactiveDetail(detail);
                setActiveProactiveSteps(steps.items ?? []);
              })}
              onSelectPluginRow={(row) => {
                if (!currentPlugin || !currentPluginState) return;
                const key = String(row[currentPlugin.rowKey] ?? "");
                void run(async () => {
                  const detail = currentPlugin.fetchDetail ? await currentPlugin.fetchDetail(row) : row;
                  setPluginState((current) => ({ ...current, [currentPlugin.id]: { ...current[currentPlugin.id], activeRowKey: key, activeDetail: detail } }));
                });
              }}
              onTogglePluginRow={(id) => {
                if (!currentPlugin) return;
                setPluginState((c) => {
                  const ps = c[currentPlugin.id];
                  if (!ps) return c;
                  const next = new Set(ps.selectedIds);
                  if (next.has(id)) next.delete(id);
                  else next.add(id);
                  return { ...c, [currentPlugin.id]: { ...ps, selectedIds: next } };
                });
              }}
              setSelectedMessageIds={setSelectedMessageIds}
              />
            </div>
          </>}
          {viewMode !== "daily" && <footer className="table-foot">
            <div>{tableMeta(viewMode, totalMessages, proactiveTotal, currentPlugin, currentPluginState, proactiveSessionFilter)}</div>
            <div className="pager">
              <button className="ghost" type="button" disabled={currentPage <= 1} onClick={() => changePage(-1)}>‹</button>
              <span>{currentPage} / {currentPageCount}</span>
              <button className="ghost" type="button" disabled={currentPage >= currentPageCount} onClick={() => changePage(1)}>›</button>
            </div>
          </footer>}
        </section>

        <aside className="detail-pane">
          <DetailPane
            viewMode={viewMode}
            activeSession={activeSession}
            activeMessage={activeMessage}
            activeProactiveDetail={activeProactiveDetail}
            activeProactiveSteps={activeProactiveSteps}
            plugin={currentPlugin}
            pluginState={currentPluginState}
            dispatch={currentDispatch}
            setProactiveSessionFilter={(key) => { setProactiveSessionFilter(key); setProactivePage(1); selectView("proactive"); }}
          />
        </aside>
      </main>
      {error && <div className="modal-backdrop" onClick={() => setError(null)}><div className="modal"><div className="modal-title">请求失败</div><p>{error}</p><div className="modal-actions"><button className="primary" type="button" onClick={() => setError(null)}>关闭</button></div></div></div>}
    </div>
  );
}

function PluginNavBody(props: {
  plugin: PluginConfig;
  pluginId: string;
  state: PluginState;
  onSetState: (updater: (s: PluginState) => PluginState) => void;
  onActivate(): void;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const getState = useEffectEvent(() => props.state);
  const filtersKey = JSON.stringify(props.state.filters);

  useEffect(() => {
    if (ref.current && props.plugin.renderNavBody) {
      const dispatch = makeDispatch(props.plugin, getState, props.onSetState, props.onActivate);
      props.plugin.renderNavBody(ref.current, dispatch);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtersKey, props.onActivate, props.plugin, props.pluginId, props.state.sortBy, props.state.sortOrder, props.state.total]);

  return <div ref={ref} />;
}

function PluginFilters(props: {
  plugin: PluginConfig;
  pluginId: string;
  state: PluginState;
  onSetState: (updater: (s: PluginState) => PluginState) => void;
  onActivate(): void;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const getState = useEffectEvent(() => props.state);
  const filtersKey = JSON.stringify(props.state.filters);

  useEffect(() => {
    if (ref.current && props.plugin.renderFilters) {
      const dispatch = makeDispatch(props.plugin, getState, props.onSetState, props.onActivate);
      props.plugin.renderFilters(ref.current, dispatch);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtersKey, props.onActivate, props.plugin, props.pluginId, props.state.sortBy, props.state.sortOrder]);

  return <div ref={ref} />;
}

function PluginTopbarAction(props: {
  plugin: PluginConfig;
  pluginId: string;
  state: PluginState;
  onSetState: (updater: (s: PluginState) => PluginState) => void;
  onActivate(): void;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const getState = useEffectEvent(() => props.state);
  const filtersKey = JSON.stringify(props.state.filters);

  useEffect(() => {
    if (ref.current && props.plugin.renderTopbarAction) {
      const dispatch = makeDispatch(props.plugin, getState, props.onSetState, props.onActivate);
      props.plugin.renderTopbarAction(ref.current, dispatch);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtersKey, props.onActivate, props.plugin, props.pluginId, props.state.sortBy, props.state.sortOrder]);

  return <div ref={ref} />;
}

function TopbarFilters(props: {
  viewMode: ViewMode;
  messageSearch: string;
  setMessageSearch(value: string): void;
  messageRole: string;
  setMessageRole(value: string): void;
  activeSessionKey: string | null;
  clearSession(): void;
  proactiveSection: string;
  proactiveSessionFilter: string;
  clearProactiveSession(): void;
  currentPlugin: PluginConfig | null;
  currentPluginState: PluginState | null;
  onSetPluginState?: (updater: (s: PluginState) => PluginState) => void;
  knowledgeVectorBackend: string;
  dailySnapshot: DailyWorkspaceSnapshot | null;
}): React.ReactElement {
  return (
    <div className="topbar-filters">
      {props.viewMode.startsWith("plugin:") && props.currentPlugin?.renderFilters && props.currentPluginState && props.onSetPluginState
        ? <PluginFilters
            plugin={props.currentPlugin}
            pluginId={props.currentPlugin.id}
            state={props.currentPluginState}
            onSetState={props.onSetPluginState}
            onActivate={() => {}}
          />
        : props.viewMode === "proactive" ? (
          <div className="filter-row">
            <div className="active-session-chip"><span>result</span><code>{proactiveSectionLabel(props.proactiveSection)}</code></div>
            {props.proactiveSessionFilter && <Chip label="session" value={props.proactiveSessionFilter} onClear={props.clearProactiveSession} />}
          </div>
        ) : props.viewMode === "knowledge" ? (
          <div className="filter-row">
            <div className="active-session-chip"><span>vector</span><code>{props.knowledgeVectorBackend || "unknown"}</code></div>
          </div>
        ) : props.viewMode === "daily" ? (
          <div className="filter-row">
            <div className="active-session-chip"><span>workspace</span><code>{props.dailySnapshot?.date ?? "loading"}</code></div>
            {props.dailySnapshot?.is_sample && <div className="active-session-chip"><span>sample</span><code>{props.dailySnapshot.sample_fallback_fields.join(", ")}</code></div>}
          </div>
        ) : (
          <div className="filter-row">
            <label className="search"><span>⌕</span><input type="text" placeholder="搜索消息内容" value={props.messageSearch} onChange={(event) => props.setMessageSearch(event.target.value.trim())} /></label>
            <select value={props.messageRole} onChange={(event) => props.setMessageRole(event.target.value)}>
              <option value="">全部 role</option><option value="user">user</option><option value="assistant">assistant</option><option value="system">system</option><option value="tool">tool</option>
            </select>
            {props.activeSessionKey && <Chip label="session" value={props.activeSessionKey} onClear={props.clearSession} />}
          </div>
        )
      }
    </div>
  );
}

function Chip(props: { label: string; value: string; onClear(): void }): React.ReactElement {
  return <div className="active-session-chip"><span>{props.label}</span><code>{props.value}</code><button type="button" onClick={props.onClear}>×</button></div>;
}

function NavGroup(props: { label: string; count: number; active: boolean; open: boolean; onToggle(): void; children: React.ReactNode }): React.ReactElement {
  return (
    <section className={`nav-group${props.active ? " active" : ""}${props.open ? " open" : ""}`}>
      <button className="nav-group-toggle" type="button" onClick={props.onToggle}>
        <span className="nav-group-caret">▸</span>
        <span className="nav-group-label">{props.label}</span>
        <span className="nav-group-count">{props.count}</span>
      </button>
      <div className={`nav-group-body${props.open ? " open" : ""}`}>
        <div className="nav-group-body-inner">{props.children}</div>
      </div>
    </section>
  );
}

function TableHead(props: {
  viewMode: ViewMode;
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  messageSortBy: string;
  messageSortOrder: SortOrder;
  proactiveSortBy: string;
  proactiveSortOrder: SortOrder;
  onSort(scope: "messages" | "proactive", key: string): void;
  onPluginSort?: (key: string) => void;
}): React.ReactElement {
  if (props.viewMode.startsWith("plugin:") && props.plugin) {
    const hasBatch = Boolean(props.plugin.batchActions?.length);
    const grid = (hasBatch ? "32px " : "") + gridTemplate(props.plugin.columns);
    const sortBy = props.pluginState?.sortBy ?? "";
    const sortOrder = props.pluginState?.sortOrder ?? "desc";
    return (
      <div className="table-head" style={{ gridTemplateColumns: grid }}>
        {hasBatch && <div />}
        {props.plugin.columns.map((col) => col.sortable && props.onPluginSort
          ? <SortHead key={col.key} label={col.label} active={sortBy === col.key} order={sortOrder} onClick={() => props.onPluginSort!(col.key)} />
          : <div key={col.key}>{col.label}</div>
        )}
      </div>
    );
  }
  if (props.viewMode === "proactive") {
    return <div className="table-head mode-proactive-ticks">
      <SortHead label="Session" active={props.proactiveSortBy === "session_key"} order={props.proactiveSortOrder} onClick={() => props.onSort("proactive", "session_key")} />
      <SortHead label="Started" active={props.proactiveSortBy === "started_at"} order={props.proactiveSortOrder} onClick={() => props.onSort("proactive", "started_at")} />
      <SortHead label="Result" active={props.proactiveSortBy === "terminal_action"} order={props.proactiveSortOrder} onClick={() => props.onSort("proactive", "terminal_action")} />
      <div>Summary</div><div />
    </div>;
  }
  return <div className="table-head mode-messages">
    <div />
    <SortHead label="Session Key" active={props.messageSortBy === "session_key"} order={props.messageSortOrder} onClick={() => props.onSort("messages", "session_key")} />
    <SortHead label="Seq" active={props.messageSortBy === "seq"} order={props.messageSortOrder} onClick={() => props.onSort("messages", "seq")} />
    <div>Content</div>
    <SortHead label="Timestamp" active={props.messageSortBy === "ts"} order={props.messageSortOrder} onClick={() => props.onSort("messages", "ts")} />
    <SortHead label="Role" active={props.messageSortBy === "role"} order={props.messageSortOrder} onClick={() => props.onSort("messages", "role")} />
    <div />
  </div>;
}

function SortHead(props: { label: string; active: boolean; order: SortOrder; onClick(): void }): React.ReactElement {
  return <button className={`table-sort-btn ${props.active ? "active" : ""}`} type="button" onClick={props.onClick}><span>{props.label}</span><span className="table-sort-arrow">{props.active ? props.order === "asc" ? "↑" : "↓" : ""}</span></button>;
}

function Rows(props: {
  viewMode: ViewMode;
  messages: MessageRow[];
  proactiveItems: ProactiveTick[];
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  selectedMessageIds: Set<string>;
  activeMessage: MessageRow | null;
  activeProactiveKey: string | null;
  onSelectMessage(item: MessageRow): void;
  onSelectProactive(item: ProactiveTick): void;
  onSelectPluginRow(row: Record<string, unknown>): void;
  onTogglePluginRow(id: string): void;
  setSelectedMessageIds(value: Set<string>): void;
}): React.ReactElement {
  if (props.viewMode.startsWith("plugin:") && props.plugin && props.pluginState) {
    const hasBatch = Boolean(props.plugin.batchActions?.length);
    const grid = (hasBatch ? "32px " : "") + gridTemplate(props.plugin.columns);
    return <>{props.pluginState.items.length ? props.pluginState.items.map((item) => {
      const key = String(item[props.plugin!.rowKey] ?? "");
      const isSelected = props.pluginState!.selectedIds.has(key);
      return <div key={key} className={`table-row ${props.pluginState!.activeRowKey === key ? "active" : ""} ${isSelected ? "selected" : ""} ${props.plugin!.rowClass?.(item) ?? ""}`} style={{ gridTemplateColumns: grid }} onClick={() => props.onSelectPluginRow(item)}>
        {hasBatch && (
          <label className="checkbox-cell" onClick={(event) => event.stopPropagation()}>
            <input type="checkbox" checked={isSelected} onChange={() => props.onTogglePluginRow(key)} />
          </label>
        )}
        {props.plugin!.columns.map((col) => {
          const cellClass = columnCellClass(col);
          if (col.renderCell) {
            return <div key={col.key} className={cellClass} title={col.rawTitle ? String(item[col.key] ?? "") : undefined} dangerouslySetInnerHTML={{ __html: col.renderCell(item[col.key], item) }} />;
          }
          return <div key={col.key} className={cellClass} title={col.rawTitle ? String(item[col.key] ?? "") : undefined}>{formatPluginCell(props.plugin!, col, item)}</div>;
        })}
      </div>;
    }) : <div className="empty-state">{props.plugin.emptyMessage || "暂无记录。"}</div>}</>;
  }
  if (props.viewMode === "proactive") {
    return <>{props.proactiveItems.map((item) => <div key={item.tick_id} className={`table-row mode-proactive-ticks ${props.activeProactiveKey === item.tick_id ? "active" : ""}`} onClick={() => props.onSelectProactive(item)}>
      <div className="cell-session mono">{formatSessionKeyForTable(item.session_key)}</div>
      <div className="cell-time">{shortTs(item.started_at)}</div>
      <div className="proactive-status-cell"><span className={`status-pill proactive-result-${proactiveResultLabel(item)}`}>{proactiveResultLabel(item)}</span><span className={`type-pill proactive-flow-${proactiveFlowLabel(item).toLowerCase()}`}>{proactiveFlowLabel(item)}</span></div>
      <div className="content-preview">{proactiveTickPreview(item)}</div>
      <div />
    </div>)}</>;
  }
  return <>{props.messages.map((item) => <div key={item.id} className={`table-row mode-messages ${props.activeMessage?.id === item.id ? "active" : ""} ${props.selectedMessageIds.has(item.id) ? "selected" : ""}`} onClick={() => props.onSelectMessage(item)}>
    <label className="checkbox-cell" onClick={(event) => event.stopPropagation()}><input type="checkbox" checked={props.selectedMessageIds.has(item.id)} onChange={(event) => toggleSet(item.id, event.target.checked, props.selectedMessageIds, props.setSelectedMessageIds)} /></label>
    <div className="cell-session mono" title={item.session_key}>{formatSessionKeyForTable(item.session_key)}</div>
    <div className="cell-seq mono">#{item.seq}</div>
    <div className="content-preview">{stripMarkdown(item.content)}</div>
    <div className="cell-time mono">{shortTs(item.ts)}</div>
    <div><span className={`role-pill ${roleClass(item.role)}`}>{item.role}</span></div>
    <div />
  </div>)}</>;
}

function DetailPane(props: {
  viewMode: ViewMode;
  activeSession: SessionRow | null;
  activeMessage: MessageRow | null;
  activeProactiveDetail: ProactiveTick | null;
  activeProactiveSteps: ProactiveStep[];
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  dispatch?: PluginDispatch;
  setProactiveSessionFilter(key: string): void;
}): React.ReactElement {
  if (props.viewMode.startsWith("plugin:") && props.plugin) {
    return <PluginDetail plugin={props.plugin} item={props.pluginState?.activeDetail ?? null} dispatch={props.dispatch} />;
  }
  if (props.viewMode === "proactive") {
    const item = props.activeProactiveDetail;
    if (!item) return <EmptyDetail text="点开 tick 后，这里会显示 proactive 执行详情和工具链。" />;
    return <div className="detail-wrap">
      <div className="detail-toolbar"><div><div className="detail-title">Tick 详情</div><div className="detail-subtext">{item.tick_id}</div></div></div>
      <button className="ghost" type="button" onClick={() => props.setProactiveSessionFilter(item.session_key)}>只看这个 session</button>
      <div className="detail-grid">
        {detailRow("session", <code>{item.session_key}</code>)}
        {detailRow("started", <code>{item.started_at}</code>)}
        {detailRow("result", <span className={`status-pill proactive-result-${proactiveResultLabel(item)}`}>{proactiveResultLabel(item)}</span>)}
        {detailRow("flow", <span className={`type-pill proactive-flow-${proactiveFlowLabel(item).toLowerCase()}`}>{proactiveFlowLabel(item)}</span>)}
      </div>
      {item.final_message && <div className="detail-block"><div className="detail-label">Final Message</div><div className="detail-content" dangerouslySetInnerHTML={{ __html: renderMarkdown(item.final_message) }} /></div>}
      <div className="detail-block"><div className="detail-label">Steps</div>{props.activeProactiveSteps.length ? props.activeProactiveSteps.map((step) => <div key={`${step.phase}-${step.step_index}`} className="tool-step"><div className="tool-step-head"><div className="tool-step-title"><span className="status-pill">step {step.step_index}</span><span className="type-pill">{step.tool_name}</span></div></div><JsonTreeBlock data={step.tool_args} /><div className="detail-content tool-result">{step.tool_result_text}</div></div>) : <div className="muted-text">没有记录到工具调用。</div>}</div>
    </div>;
  }
  if (props.activeMessage) {
    const message = props.activeMessage;
    return <div className="detail-wrap">
      <div className="detail-toolbar"><div><div className="detail-title">消息详情</div><div className="detail-subtext">{message.session_key} · #{message.seq}</div></div></div>
      <div className="detail-grid">
        {detailRow("role", <span className={`role-pill ${roleClass(message.role)}`}>{message.role}</span>)}
        {detailRow("time", <code>{message.ts}</code>)}
        {detailRow("id", <code>{message.id}</code>)}
      </div>
      <div className="detail-block"><div className="detail-label">Content</div><div className="detail-content" dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }} /></div>
      <div className="detail-block"><div className="detail-label">Extra</div><JsonTreeBlock data={message.extra} /></div>
      <div className="detail-block"><div className="detail-label">Tool Chain</div><JsonTreeBlock data={message.tool_chain} /></div>
    </div>;
  }
  if (props.activeSession) {
    const session = props.activeSession;
    return <div className="detail-wrap">
      <div className="detail-toolbar"><div><div className="detail-title">Session 详情</div><div className="detail-subtext">{session.key}</div></div></div>
      <div className="detail-grid">
        {detailRow("messages", <code>{session.message_count}</code>)}
        {detailRow("updated", <code>{session.updated_at}</code>)}
        {detailRow("last_consolidated", <code>{session.last_consolidated}</code>)}
      </div>
      <div className="detail-block"><div className="detail-label">Metadata</div><JsonTreeBlock data={session.metadata} /></div>
    </div>;
  }
  return <EmptyDetail text="点开消息、session 或 memory 后，这里会显示完整内容、字段和 JSON 信息。" />;
}

function DailyShowcasePage(props: {
  snapshot: DailyWorkspaceSnapshot | null;
  selectedDate: string;
  onSelectDate(date: string): void;
}): React.ReactElement {
  const snapshot = props.snapshot;
  if (!snapshot) {
    return (
      <main className="daily-showcase">
        <div className="showcase-loading glass-card">Loading daily workspace...</div>
      </main>
    );
  }
  const missions = snapshot.missions.filter(isRealShowcaseItem).map(cleanShowcaseMission);
  const ephemera = snapshot.ephemera.filter(isRealShowcaseItem);
  const memoryItems = snapshot.memory_items.filter(isRealShowcaseItem);
  const toolCalls = snapshot.tool_calls.filter(isRealShowcaseItem);
  const failures = snapshot.failures.filter(isRealShowcaseItem).map(cleanShowcaseMission);
  const needsApproval = snapshot.needs_approval.filter(isRealShowcaseItem).map(cleanShowcaseMission);
  const nextActions = snapshot.next_actions.filter(isRealShowcaseItem);
  const perspective = isShowcaseSampleText(snapshot.perspective) ? "今日暂无可总结的 Agent 活动" : snapshot.perspective;
  const completed = missions.filter((item) => item.status === "completed").length;
  const waiting = needsApproval.length;
  const visibleMissions = missions.slice(0, 3);
  const hiddenMissionCount = Math.max(0, missions.length - visibleMissions.length);
  return (
    <main className="daily-showcase">
      <div className="showcase-bg-layer" />
      <section className="showcase-profile glass-card">
        <div className="showcase-avatar" aria-hidden="true">
          <span>K</span>
        </div>
        <div className="showcase-profile-copy">
          <div className="daily-kicker">今日工作台</div>
          <h1>{snapshot.agent_name}</h1>
          <p>{perspective}</p>
          <div className="showcase-stats">
            <span>{snapshot.date}</span>
            <span className={`daily-status daily-status-${snapshot.status}`}>{snapshot.status}</span>
            <span>已完成 {completed} 项</span>
            <span>等待确认 {waiting} 项</span>
            {snapshot.no_real_data && <span className="sample-badge compact">No real activity yet</span>}
          </div>
        </div>
      </section>

      <section className="showcase-layout">
        <aside className="showcase-column showcase-left">
          <DailyCard title="Archive Index">
            {snapshot.archive_dates.some((item) => item.has_real_data) ? <div className="showcase-archive">
              {snapshot.archive_dates.map((item) => (
                <button key={item.date} className={props.selectedDate === item.date ? "active" : ""} type="button" onClick={() => props.onSelectDate(item.date)}>
                  <span />
                  <div>
                    <strong>{item.label}</strong>
                    <em>{item.count} logs</em>
                  </div>
                </button>
              ))}
            </div> : <ShowcaseEmpty text="暂无历史活动日期" />}
          </DailyCard>
          <DailyCard title="Memory">
            {memoryItems.length ? <TimelineList items={memoryItems.slice(0, 5).map((item) => ({
              id: item.id,
              time: item.time,
              title: item.title,
              body: stripShowcaseSamplePrefix(item.summary),
              meta: item.type,
            }))} /> : <ShowcaseEmpty text="今日暂无记忆更新" />}
          </DailyCard>
        </aside>

        <section className="showcase-column showcase-center">
          <DailyCard title="Perspective">
            <div className="perspective-note">
              <p>{perspective}</p>
              <span>{snapshot.no_real_data ? "等待今日 Agent 活动生成摘要" : snapshot.status_text}</span>
            </div>
          </DailyCard>
          <DailyCard title="Missions" className="showcase-missions">
            {visibleMissions.length ? <div className="mission-list">
              {visibleMissions.map((item) => <MissionItem key={item.id} item={item} />)}
              {hiddenMissionCount > 0 && <div className="showcase-more">+{hiddenMissionCount} more in today's log</div>}
            </div> : <ShowcaseEmpty text="今日暂无真实任务记录" />}
          </DailyCard>
        </section>

        <aside className="showcase-column showcase-right">
          <DailyCard title="Ephemera">
            {ephemera.length ? <TimelineList items={ephemera.slice(0, 7).map((item) => ({
              id: `${item.time}-${item.kind}-${item.text}`,
              time: item.time,
              title: item.kind,
              body: stripShowcaseSamplePrefix(item.text),
              meta: "agent log",
            }))} /> : <ShowcaseEmpty text="今日暂无临时活动记录" />}
          </DailyCard>
        </aside>
      </section>

      <section className="showcase-bottom">
        <DailyCard title="Next Actions">
          {nextActions.length ? <div className="compact-list">
            {nextActions.slice(0, 4).map((item) => (
              <article key={`${item.title}-${item.priority}`} className="next-action">
                <div><strong>{item.title}</strong><span>{item.priority}</span></div>
                <p>{stripShowcaseSamplePrefix(item.summary)}</p>
              </article>
            ))}
          </div> : <ShowcaseEmpty text="暂无下一步建议" />}
        </DailyCard>
        <DailyCard title="Tool Calls">
          {toolCalls.length ? <div className="tool-call-grid showcase-tool-grid">
            {toolCalls.slice(0, 6).map((item) => (
              <article key={item.id} className="tool-call-card">
                <div className="tool-call-head">
                  <span className="mono">{item.tool_name}</span>
                  <span className={`mini-status mini-status-${item.status}`}>{item.status}</span>
                </div>
                <div className="tool-call-time">{item.time}</div>
                <p>{stripShowcaseSamplePrefix(item.summary)}</p>
              </article>
            ))}
          </div> : <ShowcaseEmpty text="今日暂无工具调用" />}
        </DailyCard>
        <DailyCard title="Failures / Needs Approval">
          <div className="compact-list">
            {[...failures, ...needsApproval].length ? [...failures, ...needsApproval].slice(0, 4).map((item) => (
              <MissionItem key={`showcase-attention-${item.id}`} item={item} compact />
            )) : <ShowcaseEmpty text="今日暂无失败或待确认任务" />}
          </div>
        </DailyCard>
      </section>
    </main>
  );
}

function ShowcaseEmpty(props: { text: string }): React.ReactElement {
  return <div className="showcase-empty">{props.text}</div>;
}

function cleanShowcaseMission(item: DailyWorkspaceSnapshot["missions"][number]): DailyWorkspaceSnapshot["missions"][number] {
  return {
    ...item,
    summary: stripShowcaseSamplePrefix(item.summary),
  };
}

function stripShowcaseSamplePrefix(value: string): string {
  return value.replace(/^Sample:\s*/i, "");
}

function isRealShowcaseItem(item: unknown): boolean {
  if (!item || typeof item !== "object") return false;
  const record = item as Record<string, unknown>;
  if (record.is_sample === true) return false;
  const searchable = [
    record.source,
    record.note,
    record.label,
    record.title,
    record.summary,
    record.text,
  ].map((value) => String(value ?? ""));
  return !searchable.some(isShowcaseSampleText);
}

function isShowcaseSampleText(value: string): boolean {
  const text = value.trim().toLowerCase();
  if (!text) return false;
  return text.startsWith("sample:")
    || text.includes("sample fallback")
    || text === "sample"
    || text === "sample day"
    || text.includes("岗位监控")
    || text.includes("telegram 摘要")
    || text.includes("记忆整理")
    || text.includes("配置 vps")
    || text.includes("补充 nango")
    || text.includes("connector 凭据缺失");
}

function DailyWorkspacePanel(props: {
  snapshot: DailyWorkspaceSnapshot | null;
  selectedDate: string;
  onSelectDate(date: string): void;
}): React.ReactElement {
  const snapshot = props.snapshot;
  if (!snapshot) {
    return <div className="daily-workspace daily-loading"><div className="glass-card">Loading daily workspace...</div></div>;
  }
  return (
    <div className="daily-workspace">
      <section className="daily-hero glass-card">
        <div>
          <div className="daily-kicker">Daily Agent Workspace</div>
          <h1>{snapshot.agent_name}</h1>
          <p>{snapshot.perspective}</p>
          <div className="daily-hero-meta">
            <span className={`daily-status daily-status-${snapshot.status}`}>{snapshot.status}</span>
            <span>{snapshot.date}</span>
            <span>{snapshot.status_text}</span>
            {snapshot.is_sample && <span className="sample-badge">Sample fallback: {snapshot.sample_fallback_fields.join(", ")}</span>}
          </div>
        </div>
        <ArchiveIndex snapshot={snapshot} selectedDate={props.selectedDate} onSelectDate={props.onSelectDate} />
      </section>

      <div className="daily-grid">
        <DailyCard title="Missions" className="daily-card-wide">
          <div className="mission-list">
            {snapshot.missions.map((item) => <MissionItem key={item.id} item={item} />)}
          </div>
        </DailyCard>
        <DailyCard title="Ephemera">
          <TimelineList items={snapshot.ephemera.map((item) => ({
            id: `${item.time}-${item.kind}-${item.text}`,
            time: item.time,
            title: item.kind,
            body: item.text,
            meta: item.is_sample ? "sample" : item.source,
          }))} />
        </DailyCard>
        <DailyCard title="Memory">
          <TimelineList items={snapshot.memory_items.map((item) => ({
            id: item.id,
            time: item.time,
            title: item.title,
            body: item.summary,
            meta: `${item.type} · ${item.status}${item.is_sample ? " · sample" : ""}`,
          }))} />
        </DailyCard>
        <DailyCard title="Tool Calls" className="daily-card-wide">
          <div className="tool-call-grid">
            {snapshot.tool_calls.map((item) => (
              <article key={item.id} className="tool-call-card">
                <div className="tool-call-head">
                  <span className="mono">{item.tool_name}</span>
                  <span className={`mini-status mini-status-${item.status}`}>{item.status}</span>
                </div>
                <div className="tool-call-time">{item.time} · {item.phase || item.source}</div>
                <p>{item.summary}</p>
              </article>
            ))}
          </div>
        </DailyCard>
        <DailyCard title="Failures / Needs Approval">
          <div className="compact-list">
            {[...snapshot.failures, ...snapshot.needs_approval].length ? [...snapshot.failures, ...snapshot.needs_approval].map((item) => (
              <MissionItem key={`attention-${item.id}`} item={item} compact />
            )) : <div className="daily-empty">暂无失败或待确认事项。</div>}
          </div>
        </DailyCard>
        <DailyCard title="Next Actions">
          <div className="compact-list">
            {snapshot.next_actions.map((item) => (
              <article key={`${item.title}-${item.priority}`} className="next-action">
                <div><strong>{item.title}</strong><span>{item.priority}</span></div>
                <p>{item.summary}</p>
              </article>
            ))}
          </div>
        </DailyCard>
      </div>
    </div>
  );
}

function ArchiveIndex(props: {
  snapshot: DailyWorkspaceSnapshot;
  selectedDate: string;
  onSelectDate(date: string): void;
}): React.ReactElement {
  return (
    <div className="archive-index">
      <div className="archive-title">Archive Index</div>
      <div className="archive-days">
        {props.snapshot.archive_dates.map((item) => (
          <button key={item.date} className={props.selectedDate === item.date ? "active" : ""} type="button" onClick={() => props.onSelectDate(item.date)}>
            <span>{item.label}</span>
            <strong>{item.has_real_data ? `${item.count} logs` : "sample"}</strong>
          </button>
        ))}
      </div>
    </div>
  );
}

function DailyCard(props: { title: string; className?: string; children: React.ReactNode }): React.ReactElement {
  return <section className={`glass-card daily-card ${props.className ?? ""}`}><h2>{props.title}</h2>{props.children}</section>;
}

function MissionItem(props: { item: { title: string; status: string; time: string; summary: string; source: string; is_sample: boolean }; compact?: boolean }): React.ReactElement {
  return (
    <article className={`mission-item ${props.compact ? "compact" : ""}`}>
      <div className="mission-time">{props.item.time}</div>
      <div className="mission-body">
        <div className="mission-head">
          <strong>{props.item.title}</strong>
          <span className={`mission-status mission-status-${props.item.status}`}>{props.item.status}</span>
        </div>
        <p>{props.item.summary}</p>
        <div className="mission-source">{props.item.is_sample ? "sample fallback" : props.item.source}</div>
      </div>
    </article>
  );
}

function TimelineList(props: { items: { id: string; time: string; title: string; body: string; meta: string }[] }): React.ReactElement {
  return <div className="timeline-list">{props.items.map((item) => (
    <article key={item.id} className="timeline-item">
      <span>{item.time}</span>
      <div><strong>{item.title}</strong><p>{item.body}</p><em>{item.meta}</em></div>
    </article>
  ))}</div>;
}

function KnowledgePanel(props: {
  documents: KnowledgeDocumentRow[];
  chunks: KnowledgeChunkRow[];
  events: KnowledgeRetrievalEventRow[];
  vectorBackend: string;
  activeDocumentId: string;
  onSelectDocument(id: string): void;
}): React.ReactElement {
  return <div className="knowledge-panel">
    <section className="knowledge-column">
      <div className="detail-label">Documents · {props.vectorBackend || "unknown"}</div>
      {props.documents.length ? props.documents.map((doc) => (
        <button key={doc.id} type="button" className={`knowledge-card ${props.activeDocumentId === doc.id ? "active" : ""}`} onClick={() => props.onSelectDocument(doc.id)}>
          <div className="knowledge-title">{doc.title || doc.source_path.split("/").pop()}</div>
          <div className="knowledge-path">{doc.source_path}</div>
          <div className="knowledge-meta">{doc.file_type} · {doc.chunk_count} chunks · {shortTs(doc.updated_at)}</div>
        </button>
      )) : <div className="empty-state">暂无文档索引。</div>}
    </section>
    <section className="knowledge-column">
      <div className="detail-label">Chunks</div>
      {props.chunks.length ? props.chunks.map((chunk) => (
        <article key={chunk.id} className="knowledge-card">
          <div className="knowledge-title">#{chunk.chunk_index} {chunk.heading_path || chunk.title}</div>
          <div className="knowledge-meta">行 {chunk.line_start}-{chunk.line_end} · {chunk.token_count} tokens</div>
          <div className="knowledge-snippet">{chunk.text}</div>
        </article>
      )) : <div className="empty-state">暂无 chunk。</div>}
    </section>
    <section className="knowledge-column">
      <div className="detail-label">Retrieval Events</div>
      {props.events.length ? props.events.map((event) => (
        <article key={event.id} className="knowledge-card">
          <div className="knowledge-title">{event.query}</div>
          <div className="knowledge-meta">{shortTs(event.created_at)} · retrieved {event.retrieved_chunk_ids.length} · injected {event.injected_chunk_ids.length}</div>
          {event.trace_id && <div className="knowledge-path">trace: {event.trace_id}</div>}
        </article>
      )) : <div className="empty-state">暂无检索记录。</div>}
    </section>
  </div>;
}

function EmptyDetail(props: { text: string }): React.ReactElement {
  return <div className="detail-empty"><div className="detail-empty-title">详情</div><div className="detail-empty-text">{props.text}</div></div>;
}

function detailRow(label: string, value: React.ReactNode): React.ReactElement {
  return <div className="detail-row"><div className="detail-row-label">{label}</div><div className="detail-row-val">{value}</div></div>;
}

function JsonTreeBlock(props: { data: unknown }): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const payload = JSON.stringify(props.data ?? null);

  useEffect(() => {
    if (!ref.current) return;
    ref.current.innerHTML = jvPlaceholder(props.data);
    attachJsonViewers(ref.current);
  }, [payload, props.data]);

  return <div ref={ref} />;
}

function toggleSet(id: string, checked: boolean, source: Set<string>, update: (value: Set<string>) => void): void {
  const next = new Set(source);
  if (checked) next.add(id);
  else next.delete(id);
  update(next);
}

function gridTemplate(columns: DashboardColumn[]): string {
  return columns.map((col) => col.flex ? "1fr" : col.width ? `${col.width}px` : "auto").join(" ");
}

function formatPluginCell(plugin: PluginConfig, column: DashboardColumn, item: Record<string, unknown>): string {
  const value = item[column.key];
  const formatter = plugin.formatters?.[column.fmt || ""] ?? (window as Window & { KotarouDashboard?: { _formatters: Record<string, (value: unknown, item?: Record<string, unknown>) => string> } }).KotarouDashboard?._formatters[column.fmt || "text"];
  return formatter ? formatter(value, item) : String(value ?? "");
}

function columnCellClass(column: DashboardColumn): string {
  const classes = [column.cellClass ?? ""];
  if (column.align === "right") classes.push("align-right");
  return classes.filter(Boolean).join(" ");
}

function tableMeta(viewMode: ViewMode, totalMessages: number, proactiveTotal: number, plugin: PluginConfig | null, pluginState: PluginState | null, proactiveSessionFilter: string): string {
  if (plugin && pluginState) return plugin.countTitle ? plugin.countTitle(pluginState.total) : `共 ${pluginState.total} 条`;
  if (viewMode === "daily") return "Daily Agent Workspace";
  if (viewMode === "knowledge") return "Document RAG 知识库";
  if (viewMode === "proactive") return proactiveSessionFilter ? `共 ${proactiveTotal} 条 tick · session: ${proactiveSessionFilter}` : `共 ${proactiveTotal} 条 tick`;
  return `共 ${totalMessages} 条`;
}

function totalSessionMessages(sessions: SessionRow[]): number {
  return sessions.reduce((sum, session) => sum + (session.message_count || 0), 0);
}

function proactiveSectionCount(section: string, overview: ProactiveOverview | null): number {
  if (!overview) return 0;
  if (section === "all") return overview.counts.tick_logs ?? 0;
  if (section === "drift" || section === "proactive") return overview.flow_counts[section] ?? 0;
  return overview.result_counts[section] ?? 0;
}

function viewLabel(viewMode: ViewMode, plugin: PluginConfig | null): string {
  if (plugin) return plugin.viewLabel || plugin.label;
  if (viewMode === "daily") return "daily";
  if (viewMode === "knowledge") return "knowledge";
  if (viewMode === "proactive") return "proactive";
  return "messages";
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
