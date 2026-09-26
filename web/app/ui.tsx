"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

type Health = {
  llm_provider: string;
  llm_model: string;
  llm_configured: boolean;
  embeddings_configured: boolean;
  rerank_configured: boolean;
  google_configured: boolean;
  drive_connected: boolean;
  oauth_start?: string;
};

type Source = {
  id: string;
  type: string;
  uri: string;
  title: string;
  status: string;
  error: string;
  page_count: number;
  chunk_count: number;
};

type Questionnaire = {
  id: string;
  filename: string;
  status: string;
  confirmed: boolean;
  question_count: number;
  answered_count: number;
  error: string;
};

type OrgInfo = {
  id: string;
  name: string;
  confirmed?: boolean;
  preferences: { tone?: string; never_commit?: string };
};

type Chat = {
  id: string;
  title: string;
  project_id?: string;
  started_at?: string | null;
};

function chatLabel(chat: Chat) {
  if (chat.title && chat.title !== "New chat") return chat.title;
  if (chat.started_at) return `Chat ${chat.started_at.slice(0, 16).replace("T", " ")}`;
  return "Untitled chat";
}

type Screen = "org" | "projects" | "workspace";

type ProjectInfo = {
  id: string;
  name: string;
  customer: string;
  status: string;
};

type RunInfo = {
  id: string;
  status: string;
  latency_ms?: number;
  model_id: string;
  token_usage: { input?: number; output?: number; cost_usd?: number };
  retrieved: { title?: string; score?: number }[];
  budget_reason?: string;
  validation?: { status?: string; citations?: number };
};

type Session = {
  id: string;
  project_id?: string;
  intake_state: string;
  sources: Source[];
  questionnaires?: Questionnaire[];
};

type Citation = {
  title: string;
  uri: string;
  locator: string;
  quote: string;
};

type Answer = {
  id: string;
  question_id?: string | null;
  question_text: string;
  status: string;
  origin: string;
  answer_text: string;
  short_answer?: string | null;
  confidence: number;
  citations: Citation[];
  gaps: string[];
  conflicts: string[];
  run_id?: string;
};

type QuestionRow = {
  id: string;
  ordinal: number;
  text: string;
  category: string;
  constraints: string;
  answer: Answer | null;
};

type MemoryItem = {
  id: string;
  question_text: string;
  answer_text: string;
  valid_until: string | null;
};

const API = "/backend";

function isDriveUri(uri: string): boolean {
  const lowered = uri.toLowerCase();
  return lowered.includes("drive.google.com") || lowered.includes("docs.google.com");
}

function asList(value: unknown): string[] {
  if (Array.isArray(value)) return value.map((item) => String(item ?? "").trim()).filter(Boolean);
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
}

function asCitations(value: unknown): Citation[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is Citation => Boolean(item) && typeof item === "object");
}

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [url, setUrl] = useState("");
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [thread, setThread] = useState<Answer[]>([]);
  const [questions, setQuestions] = useState<QuestionRow[]>([]);
  const [memory, setMemory] = useState<MemoryItem[]>([]);
  const [includeDrafts, setIncludeDrafts] = useState(false);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [driveNotice, setDriveNotice] = useState("");
  const [org, setOrg] = useState<OrgInfo | null>(null);
  const [projects, setProjects] = useState<ProjectInfo[]>([]);
  const [projectId, setProjectId] = useState("");
  const [runsByChat, setRunsByChat] = useState<Record<string, RunInfo[]>>({});
  const [orgName, setOrgName] = useState("");
  const [tone, setTone] = useState("");
  const [neverCommit, setNeverCommit] = useState("");
  const [projectName, setProjectName] = useState("");
  const [customer, setCustomer] = useState("");
  const [screen, setScreen] = useState<Screen>("org");
  const [projectSources, setProjectSources] = useState<Source[]>([]);
  const [chats, setChats] = useState<Chat[]>([]);

  const indexing = projectSources.some((s) => ["pending", "fetching", "indexing"].includes(s.status));
  const batchRunning = session?.questionnaires?.some((q) => q.status === "running");

  const loadSession = useCallback(async (id: string) => {
    const res = await fetch(`${API}/sessions/${id}`);
    if (!res.ok) throw new Error("Could not load session");
    const data = (await res.json()) as Session;
    setSession(data);
    return data;
  }, []);

  const loadAnswers = useCallback(async (id: string) => {
    const res = await fetch(`${API}/sessions/${id}/answers`);
    if (!res.ok) return;
    const data = (await res.json()) as { answers: Answer[] };
    setThread(data.answers.filter((a) => !a.question_id));
    setEdits((prev) => {
      const next = { ...prev };
      for (const answer of data.answers) {
        if (next[answer.id] === undefined) next[answer.id] = answer.answer_text;
      }
      return next;
    });
  }, []);

  const loadQuestions = useCallback(async (id: string) => {
    const res = await fetch(`${API}/sessions/${id}/questions`);
    if (!res.ok) return;
    const data = (await res.json()) as { questions: QuestionRow[] };
    setQuestions(data.questions);
  }, []);

  const loadRuns = useCallback(async (id: string) => {
    const res = await fetch(`${API}/sessions/${id}/runs`);
    if (!res.ok) return;
    const data = (await res.json()) as { runs: RunInfo[] };
    setRunsByChat((prev) => ({ ...prev, [id]: data.runs }));
  }, []);

  const loadProjectBundle = useCallback(async (id: string) => {
    const [sourceRes, chatRes] = await Promise.all([
      fetch(`${API}/projects/${id}/sources`),
      fetch(`${API}/projects/${id}/sessions`),
    ]);
    let sessions: Chat[] = [];
    if (sourceRes.ok) {
      const data = (await sourceRes.json()) as { sources: Source[] };
      setProjectSources(data.sources);
    }
    if (chatRes.ok) {
      const data = (await chatRes.json()) as { sessions: Chat[] };
      sessions = data.sessions;
      setChats(sessions);
      await Promise.all(sessions.map((chat) => loadRuns(chat.id)));
    }
    return sessions;
  }, [loadRuns]);

  const loadMemory = useCallback(async () => {
    const res = await fetch(`${API}/memory`);
    if (!res.ok) return;
    const data = (await res.json()) as { items: MemoryItem[] };
    setMemory(data.items);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const params = new URLSearchParams(window.location.search);
      const driveStatus = params.get("drive");
      if (driveStatus === "error") {
        setError(params.get("reason") || "Google Drive login failed. Try Connect Google Drive again.");
      } else if (driveStatus === "connected") {
        setDriveNotice(
          "Google Drive is signed in. That only stores a read-only token. Open a project, paste a Drive file or folder URL, and click Index.",
        );
      }
      const h = await fetch(`${API}/health`).then((r) => r.json());
      if (!cancelled) setHealth(h);
      const orgRes = await fetch(`${API}/org`);
      if (orgRes.ok) {
        const orgData = (await orgRes.json()) as OrgInfo;
        if (!cancelled) {
          setOrg(orgData);
          setOrgName(orgData.name);
          setTone(orgData.preferences?.tone || "");
          setNeverCommit(orgData.preferences?.never_commit || "");
        }
      }
      const projectRes = await fetch(`${API}/projects`);
      const projectData = projectRes.ok ? ((await projectRes.json()) as { projects: ProjectInfo[] }) : { projects: [] };
      if (!cancelled) setProjects(projectData.projects);
      await loadMemory();
      if (!cancelled) setScreen("org");
    })().catch((err: Error) => setError(err.message));
    return () => {
      cancelled = true;
    };
  }, [loadMemory]);

  useEffect(() => {
    if (screen !== "workspace" || !projectId || !(indexing || batchRunning)) return;
    const timer = window.setInterval(() => {
      void loadProjectBundle(projectId);
      if (session) {
        void loadSession(session.id);
        void loadQuestions(session.id);
        void loadAnswers(session.id);
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [screen, projectId, session, indexing, batchRunning, loadProjectBundle, loadSession, loadQuestions, loadAnswers]);

  async function addSource(uri: string) {
    if (!projectId) return;
    if (isDriveUri(uri) && !health?.drive_connected) {
      setError("Connect Google Drive first, then paste the same file or folder URL and click Index. Connecting does not pull files by itself.");
      return;
    }
    setBusy(true);
    setError("");
    setDriveNotice("");
    try {
      const res = await fetch(`${API}/projects/${projectId}/sources`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "url", uri, title: "" }),
      });
      if (!res.ok) throw new Error(await res.text());
      await loadProjectBundle(projectId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Source failed");
    } finally {
      setBusy(false);
    }
  }

  async function upload(file: File) {
    if (!projectId) return;
    const body = new FormData();
    body.append("file", file);
    setBusy(true);
    try {
      const res = await fetch(`${API}/projects/${projectId}/sources/upload`, { method: "POST", body });
      if (!res.ok) throw new Error(await res.text());
      await loadProjectBundle(projectId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function reindex(sourceId: string) {
    if (!projectId) return;
    await fetch(`${API}/projects/${projectId}/sources/${sourceId}/reindex`, { method: "POST" });
    await loadProjectBundle(projectId);
  }

  async function detach(sourceId: string) {
    if (!projectId) return;
    await fetch(`${API}/projects/${projectId}/sources/${sourceId}`, { method: "DELETE" });
    await loadProjectBundle(projectId);
  }

  async function openChat(chat: Chat) {
    window.localStorage.setItem("rfi_session", chat.id);
    window.localStorage.setItem(`rfi_session:${projectId}`, chat.id);
    await loadSession(chat.id);
    await loadAnswers(chat.id);
    await loadQuestions(chat.id);
    await loadRuns(chat.id);
  }

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!projectId || !question.trim()) return;
    let chat = session;
    if (!chat) {
      const created = await fetch(`${API}/projects/${projectId}/sessions`, { method: "POST" }).then((r) => r.json());
      chat = created as Session;
      setSession(chat);
      setChats((prev) => [{ id: created.id, title: created.title || "New chat", project_id: projectId }, ...prev]);
      window.localStorage.setItem("rfi_session", created.id);
    }
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API}/sessions/${chat.id}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (res.status === 409) throw new Error("Index sources or skip first.");
      if (!res.ok) throw new Error(await res.text());
      const answer = (await res.json()) as Answer;
      setThread((prev) => [...prev, answer]);
      setEdits((prev) => ({ ...prev, [answer.id]: answer.answer_text }));
      setQuestion("");
      await loadRuns(chat.id);
      await loadProjectBundle(projectId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ask failed");
    } finally {
      setBusy(false);
    }
  }

  async function saveEdit(answer: Answer) {
    if (!session) return;
    const text = edits[answer.id] ?? answer.answer_text;
    await fetch(`${API}/sessions/${session.id}/answers/${answer.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer_text: text }),
    });
  }

  async function review(answer: Answer, decision: "approve" | "reject") {
    if (!session) return;
    if (decision === "approve") await saveEdit(answer);
    const res = await fetch(`${API}/sessions/${session.id}/answers/${answer.id}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision }),
    });
    if (!res.ok) {
      setError(await res.text());
      return;
    }
    const updated = (await res.json()) as Answer;
    setThread((prev) => prev.map((item) => (item.id === updated.id ? updated : item)));
    await loadQuestions(session.id);
    await loadMemory();
  }

  async function uploadQuestionnaire(file: File) {
    if (!session) return;
    const body = new FormData();
    body.append("file", file);
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API}/sessions/${session.id}/questionnaires`, { method: "POST", body });
      if (res.status === 409) throw new Error("Index sources or skip first.");
      if (!res.ok) throw new Error(await res.text());
      const row = (await res.json()) as Questionnaire;
      await loadSession(session.id);
      await loadQuestions(session.id);
      if (row.status === "parsed") {
        await runQuestionnaire(row.id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Questionnaire failed");
    } finally {
      setBusy(false);
    }
  }

  async function runQuestionnaire(id: string, confirm = false) {
    if (!session) return;
    const res = await fetch(`${API}/sessions/${session.id}/questionnaires/${id}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm }),
    });
    if (!res.ok) {
      setError(await res.text());
      return;
    }
    await loadSession(session.id);
    await loadQuestions(session.id);
  }

  async function newSession() {
    if (!projectId) return;
    const created = await fetch(`${API}/projects/${projectId}/sessions`, { method: "POST" }).then((r) => r.json());
    window.localStorage.setItem("rfi_session", created.id);
    window.localStorage.setItem(`rfi_session:${projectId}`, created.id);
    setSession({ ...created, sources: projectSources, questionnaires: [] });
    await loadProjectBundle(projectId);
    setThread([]);
    setQuestions([]);
    setQuestion("");
  }

  async function saveOrg() {
    if (!orgName.trim()) {
      setError("Organization name is required.");
      return;
    }
    const res = await fetch(`${API}/org`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: orgName.trim(), preferences: { tone, never_commit: neverCommit } }),
    });
    if (!res.ok) {
      setError(await res.text());
      return;
    }
    setOrg((await res.json()) as OrgInfo);
    setError("");
    setScreen("projects");
  }

  async function createProject(event: FormEvent) {
    event.preventDefault();
    const name = projectName.trim();
    if (!name) return;
    const res = await fetch(`${API}/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, customer }),
    });
    if (!res.ok) {
      setError(await res.text());
      return;
    }
    const created = (await res.json()) as ProjectInfo;
    setProjects((prev) => [...prev, created]);
    setProjectName("");
    setCustomer("");
    await openProject(created.id);
  }

  async function openProject(id: string) {
    setProjectId(id);
    window.localStorage.setItem("rfi_project", id);
    setScreen("workspace");
    setSession(null);
    setThread([]);
    setQuestions([]);
    setRunsByChat({});
    const sessions = await loadProjectBundle(id);
    const saved = window.localStorage.getItem(`rfi_session:${id}`);
    const openSaved = async (sessionId: string) => {
      const loaded = await loadSession(sessionId);
      if (loaded.project_id && loaded.project_id !== id) {
        throw new Error("Chat belongs to another project");
      }
      window.localStorage.setItem("rfi_session", sessionId);
      window.localStorage.setItem(`rfi_session:${id}`, sessionId);
      await loadAnswers(sessionId);
      await loadQuestions(sessionId);
      await loadRuns(sessionId);
    };
    if (saved) {
      try {
        await openSaved(saved);
        return;
      } catch {
        window.localStorage.removeItem(`rfi_session:${id}`);
        setSession(null);
      }
    }
    if (sessions[0]) await openSaved(sessions[0].id);
  }

  async function deleteMemory(id: string) {
    const res = await fetch(`${API}/memory/${id}`, { method: "DELETE" });
    if (!res.ok) {
      setError(await res.text());
      return;
    }
    await loadMemory();
  }

  const statusLine = useMemo(() => {
    if (!health) return "Starting…";
    const bits = [
      `${health.llm_provider}/${health.llm_model}`,
      health.llm_configured ? "LLM ready" : "LLM key missing",
      health.rerank_configured ? "rerank on" : "RRF only",
      health.google_configured ? (health.drive_connected ? "Drive connected" : "Drive OAuth ready") : "Drive client not set",
    ];
    return bits.join(" · ");
  }, [health]);

  const latestQuestionnaire = session?.questionnaires?.[0];
  const draftQuery = includeDrafts ? "?include_drafts=true" : "";
  const chatSections = (() => {
    const list = [...chats];
    if (session && !list.some((chat) => chat.id === session.id)) {
      list.unshift({ id: session.id, title: "New chat", project_id: projectId });
    }
    if (!session) return list;
    return [...list.filter((chat) => chat.id === session.id), ...list.filter((chat) => chat.id !== session.id)];
  })();

  return (
    <main className="shell">
      <div className="eyebrow">RFI Agent</div>
      <h1>{screen === "org" ? "Organization" : screen === "projects" ? "Projects" : projects.find((item) => item.id === projectId)?.name || "Project"}</h1>
      <p className="lead">
        {screen === "org"
          ? "Save the organization first. Approved answers are shared by every project in it."
          : screen === "projects"
            ? "Each project keeps one document index. Chats in that project all ask against it."
            : "Chats in this project share the index above. A new chat does not delete documents."}
      </p>
      <p className="status">{statusLine}</p>
      {driveNotice ? <p className="banner ok">{driveNotice}</p> : null}
      {error ? <p className="banner">{error}</p> : null}

      {screen === "org" ? (
        <section className="card">
          <h2>Organization information</h2>
          <label htmlFor="org-name">Name</label>
          <input id="org-name" type="text" value={orgName} onChange={(e) => setOrgName(e.target.value)} placeholder="Company name" />
          <label htmlFor="tone">Tone</label>
          <input id="tone" type="text" value={tone} onChange={(e) => setTone(e.target.value)} placeholder="Concise, no marketing language" />
          <label htmlFor="never">Never commit to</label>
          <input id="never" type="text" value={neverCommit} onChange={(e) => setNeverCommit(e.target.value)} placeholder="pricing, legal warranties" />
          <div className="row">
            <button type="button" onClick={() => void saveOrg()} disabled={!org || !orgName.trim()}>
              Save and continue
            </button>
          </div>
        </section>
      ) : null}

      {screen === "projects" ? (
        <>
          <section className="card">
            <div className="chat-head">
              <h2>Your projects</h2>
              <button type="button" className="secondary" onClick={() => setScreen("org")}>
                Edit organization
              </button>
            </div>
            {projects.length ? (
              projects.map((item) => (
                <div key={item.id} className="project-row">
                  <div>
                    <p>
                      <strong>{item.name}</strong>
                    </p>
                    {item.customer ? <p className="meta">{item.customer}</p> : null}
                  </div>
                  <button type="button" onClick={() => void openProject(item.id)}>
                    Open
                  </button>
                </div>
              ))
            ) : (
              <p className="status">No projects yet. Create one below, then open it from this list.</p>
            )}
          </section>
          <section className="card">
            <h2>Create a project</h2>
            <form onSubmit={(event) => void createProject(event)}>
              <label htmlFor="new-project">Name</label>
              <input id="new-project" type="text" value={projectName} onChange={(e) => setProjectName(e.target.value)} placeholder="Security RFI" />
              <label htmlFor="customer">Customer</label>
              <input id="customer" type="text" value={customer} onChange={(e) => setCustomer(e.target.value)} placeholder="Customer" />
              <div className="row">
                <button type="submit" disabled={!projectName.trim()}>
                  Create project
                </button>
              </div>
            </form>
          </section>
        </>
      ) : null}

      {screen === "workspace" ? (
        <>
      <div className="row">
        <button type="button" className="secondary" onClick={() => setScreen("projects")}>
          All projects
        </button>
      </div>
      <section className="card">
        <h2>Project index</h2>
        <p className="status">
          Documents indexed here are shared by every chat in this project. Connecting Drive only signs you in. Paste a file
          or folder URL and click Index.
        </p>
        <SourceForm
          url={url}
          setUrl={setUrl}
          busy={busy}
          onIndex={addSource}
          sessionId={projectId}
          onUpload={upload}
          oauthStart={health?.oauth_start}
          healthReady={Boolean(health)}
          googleConfigured={health?.google_configured}
          driveConnected={health?.drive_connected}
        />
      </section>

      <section className="card">
          <h2>Sources</h2>
          {projectSources.length ? null : <p className="status">No documents indexed yet.</p>}
          {projectSources.map((source) => (
            <div key={source.id} className="source-row">
              <p className="status">
                {source.title || source.uri} — {source.status}
                {source.page_count ? ` · ${source.page_count} pages` : ""}
                {source.chunk_count ? ` · ${source.chunk_count} chunks` : ""}
                {source.error ? ` · ${source.error}` : ""}
              </p>
              {source.status !== "detached" ? (
                <div className="row">
                  <button className="secondary" type="button" onClick={() => void reindex(source.id)} disabled={busy || indexing}>
                    Reindex
                  </button>
                  <button className="secondary" type="button" onClick={() => void detach(source.id)} disabled={busy}>
                    Detach
                  </button>
                </div>
              ) : null}
            </div>
          ))}
        </section>

      <section className="card">
          <h2>Questionnaire</h2>
          <p className="status">
            {session
              ? "Upload CSV, XLSX, or DOCX for this chat. More than 20 questions asks for confirmation before drafting."
              : chats.length
                ? "Open a chat to upload a questionnaire."
                : "Create a chat before uploading a questionnaire."}
          </p>
          {session ? (
          <input
            type="file"
            accept=".csv,.xlsx,.docx"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void uploadQuestionnaire(file);
            }}
          />
          ) : null}
          {latestQuestionnaire ? (
            <p className="status">
              {latestQuestionnaire.filename} — {latestQuestionnaire.status}
              {` · ${latestQuestionnaire.answered_count}/${latestQuestionnaire.question_count}`}
              {latestQuestionnaire.error ? ` · ${latestQuestionnaire.error}` : ""}
            </p>
          ) : null}
          {latestQuestionnaire?.status === "needs_confirm" ? (
            <button type="button" onClick={() => void runQuestionnaire(latestQuestionnaire.id, true)}>
              Confirm and draft {latestQuestionnaire.question_count} questions
            </button>
          ) : null}
          {latestQuestionnaire && ["parsed", "partial"].includes(latestQuestionnaire.status) ? (
            <button type="button" className="secondary" onClick={() => void runQuestionnaire(latestQuestionnaire.id)}>
              {latestQuestionnaire.status === "partial" ? "Resume drafting" : "Draft answers"}
            </button>
          ) : null}
          {questions.length ? (
            <table className="table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Question</th>
                  <th>Status</th>
                  <th>Answer</th>
                </tr>
              </thead>
              <tbody>
                {questions.map((row) => (
                  <tr key={row.id}>
                    <td>{row.ordinal}</td>
                    <td>
                      {row.text}
                      {row.category ? <div className="status">{row.category}</div> : null}
                    </td>
                    <td>{row.answer?.status || "queued"}</td>
                    <td>
                      {row.answer?.short_answer || row.answer?.answer_text || "—"}
                      <AnswerTrace run={session ? (runsByChat[session.id] || []).find((run) => run.id === row.answer?.run_id) : undefined} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </section>

      <section className="card">
        <div className="chat-head">
          <h2>Chats</h2>
          <button type="button" onClick={() => void newSession()}>
            New chat
          </button>
        </div>
        <p className="status">Each chat asks against this project’s index. A trace sits on the answer it produced.</p>
      </section>
      {chatSections.map((chat) => {
        const open = chat.id === session?.id;
        return (
          <section className="card" key={chat.id}>
            <div className="chat-head">
              <h2>{chatLabel(chat)}</h2>
              {open ? null : (
                <button type="button" className="secondary" onClick={() => void openChat(chat)}>
                  Open
                </button>
              )}
            </div>
            {open ? (
              <>
                <h3 className="subhead">Ask in this chat</h3>
                <form onSubmit={ask}>
                  <textarea rows={3} value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a question about this project's documents…" />
                  <div className="row">
                    <button type="submit" disabled={busy || indexing || !question.trim()}>
                      {indexing ? "Indexing…" : "Ask"}
                    </button>
                  </div>
                </form>
                {thread.length ? (
                  <div className="chat">
                    {thread.map((item) => (
                      <article key={item.id} className="bubble">
                        <div className="eyebrow">{item.status} · {item.origin}{item.short_answer ? ` · ${item.short_answer}` : ""}</div>
                        <p>
                          <strong>Q.</strong> {item.question_text}
                        </p>
                        <textarea
                          rows={4}
                          value={edits[item.id] ?? item.answer_text}
                          onChange={(e) => setEdits((prev) => ({ ...prev, [item.id]: e.target.value }))}
                        />
                        {asCitations(item.citations).map((cite, i) => (
                          <div className="cite" key={`${item.id}-${i}`}>
                            {cite.uri ? <a href={cite.uri}>{cite.title || cite.uri}</a> : cite.title} {cite.locator}
                            {cite.quote ? <div>“{cite.quote}”</div> : null}
                          </div>
                        ))}
                        {asList(item.gaps).length ? <p className="status">Gaps: {asList(item.gaps).join("; ")}</p> : null}
                        {asList(item.conflicts).length ? <p className="status">Conflicts: {asList(item.conflicts).join("; ")}</p> : null}
                        <AnswerTrace run={(runsByChat[chat.id] || []).find((run) => run.id === item.run_id)} />
                        <div className="row">
                          <button className="secondary" type="button" onClick={() => void review(item, "approve")}>
                            Approve into memory
                          </button>
                          <button className="secondary" type="button" onClick={() => void review(item, "reject")}>
                            Reject
                          </button>
                        </div>
                      </article>
                    ))}
                  </div>
                ) : null}
                <h3 className="subhead">Export</h3>
                <label className="check">
                  <input type="checkbox" checked={includeDrafts} onChange={(e) => setIncludeDrafts(e.target.checked)} />
                  Include unreviewed drafts
                </label>
                <div className="row">
                  <a className="btn secondary" href={`${API}/sessions/${chat.id}/export.json${draftQuery}`}>
                    Download JSON
                  </a>
                  <a className="btn secondary" href={`${API}/sessions/${chat.id}/export.xlsx${draftQuery}`}>
                    Download XLSX
                  </a>
                </div>
              </>
            ) : null}
          </section>
        );
      })}
        </>
      ) : null}

      {memory.length && screen !== "org" ? (
        <section className="card">
          <h2>Approved memory</h2>
          <p className="status">Shared across projects in {org?.name || "this organization"}.</p>
          {memory.slice(0, 8).map((item) => (
            <div key={item.id} className="source-row">
              <p className="status">
                {item.question_text}
                {item.valid_until ? ` · until ${item.valid_until.slice(0, 10)}` : ""}
              </p>
              <button className="secondary" type="button" onClick={() => void deleteMemory(item.id)}>
                Remove from memory
              </button>
            </div>
          ))}
        </section>
      ) : null}
    </main>
  );
}

function AnswerTrace({ run }: { run?: RunInfo }) {
  if (!run) return null;
  const titles = [...new Set((run.retrieved || []).map((hit) => hit.title || "chunk"))];
  const shown = titles.slice(0, 6);
  return (
    <div className="trace">
      <p className="status">
        Trace · {run.status}
        {run.model_id ? ` · ${run.model_id}` : ""}
        {run.latency_ms != null ? ` · ${run.latency_ms} ms` : ""}
        {run.token_usage?.input != null ? ` · ${run.token_usage.input} in` : ""}
        {run.token_usage?.cost_usd != null ? ` · $${run.token_usage.cost_usd}` : ""}
        {run.budget_reason ? ` · budget: ${run.budget_reason}` : ""}
      </p>
      {shown.length ? (
        <p className="status">
          Retrieved: {shown.join(", ")}
          {titles.length > shown.length ? ` · +${titles.length - shown.length} more` : ""}
        </p>
      ) : null}
    </div>
  );
}

function SourceForm({
  url,
  setUrl,
  busy,
  onIndex,
  sessionId,
  onSkip,
  onUpload,
  oauthStart,
  healthReady,
  googleConfigured,
  driveConnected,
}: {
  url: string;
  setUrl: (value: string) => void;
  busy: boolean;
  onIndex: (uri: string) => void;
  sessionId?: string;
  onSkip?: () => void;
  onUpload: (file: File) => void;
  oauthStart?: string;
  healthReady?: boolean;
  googleConfigured?: boolean;
  driveConnected?: boolean;
}) {
  const driveHref = `${oauthStart || "http://localhost:8000/oauth/google/start"}?session_id=${sessionId || ""}`;
  return (
    <>
      <div style={{ marginTop: 8 }}>
        <label htmlFor="url">Website or Google Drive URL</label>
        <div className="row">
          <input
            id="url"
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://docs.example.com/ or https://drive.google.com/drive/folders/…"
            onKeyDown={(e) => {
              if (e.key === "Enter" && url.trim()) {
                e.preventDefault();
                onIndex(url.trim());
                setUrl("");
              }
            }}
          />
          <button
            disabled={busy || !url.trim()}
            onClick={() => {
              const value = url.trim();
              onIndex(value);
              setUrl("");
            }}
          >
            Index
          </button>
        </div>
        <p className="status">
          Public HTTPS pages are fetched directly. For Drive, sign in below, then paste a file or folder link here and
          Index. Connecting Drive does not import anything on its own.
        </p>
      </div>
      <div className="row" style={{ marginTop: 16 }}>
        {healthReady && googleConfigured ? (
          <a className="btn secondary" href={driveHref}>
            {driveConnected ? "Reconnect Google Drive" : "Connect Google Drive"}
          </a>
        ) : null}
        {healthReady && !googleConfigured ? (
          <span className="status" style={{ marginTop: 0 }}>
            Drive client is not configured in .env, so OAuth is unavailable.
          </span>
        ) : null}
        {onSkip ? (
          <button className="secondary" onClick={onSkip} disabled={busy} type="button">
            Skip
          </button>
        ) : null}
      </div>
      {healthReady && googleConfigured ? (
        <p className="status">
          {driveConnected
            ? "Drive is signed in. Paste a Drive file or folder URL above and click Index."
            : "Connect Google Drive to authorize read-only access. After you return, paste the Drive URL and Index."}
        </p>
      ) : null}
      <div style={{ marginTop: 16 }}>
        <label htmlFor="file">Or upload a file from this computer</label>
        <input
          id="file"
          type="file"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUpload(file);
          }}
        />
      </div>
    </>
  );
}
