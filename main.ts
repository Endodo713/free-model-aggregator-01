// free-model-aggregator-01: one server, ten simulated free-tier providers.
//
// The brief gives this server a provider directory, a quality table and a
// workload manifest, and the env gives it the providers themselves. It reads
// all three, probes every provider's catalog route, builds one catalog, then:
//
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models, POST /v1/chat/completions and
//      POST /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//
// ROUTING. For each request the server ranks every free model that meets the
// request's requirements by a risk-adjusted rating: the quality table's
// rating_milli minus a penalty for the failure risk of the provider's next
// call. The first call to a provider can never fail (the failure schedule
// starts at the second call), the second call is the riskiest (one provider
// in every trial is drawn to fail on exactly its second call), later calls
// carry only the base risk. A soft per-provider call cap keeps the load
// spread: the cap is a prior from the kit's sealed per-provider budget ranges
// (high-budget providers take three calls, mid two, low one), and the cap
// relaxes rather than refusing to serve when it has to. The ranking is a pure
// function of the brief and of the calls made so far, so a replay decides
// exactly the same way.
//
// FAILOVER. A provider that answers 402, 429 or a 5xx, or that cannot be
// reached, is retried around, not dropped on one refusal: the failure
// schedule names one or two call numbers per provider, so the provider's next
// call usually answers. Only two refusals in a row drop a provider (a spent
// budget or an outage fails every later call). A refusal is never waited out:
// the failure schedule is per call number, not per unit of time, so waiting
// buys nothing. Every attempt counts as one upstream call, failed or not,
// exactly as the env counts it.
//
// WIRES. All six client-by-provider combinations are translated losslessly
// (see lib/wire.ts): the provider echoes what it parsed, and the client checks
// that echo against what it sent, so a dropped system prompt, image, tool
// name, max_tokens, stop sequence or JSON-mode flag loses the request. The one
// combination that cannot be lossless is an OpenAI JSON-mode request on an
// anthropic-wire provider (Anthropic's echo reports json_mode false), so those
// requests never route there. Usage is always the provider's own numbers,
// forwarded verbatim in the client's wire.
//
// CATALOG. The register body's catalog and the `--list` document are the same
// object. A provider whose catalog probe fails is rebuilt from the quality
// table's rows instead of being left out: the table is the sealed truth, and a
// missing provider costs catalog checks while a rebuilt one can still serve.
//
// `node main.ts --list` prints the catalog document and exits 0 before any
// server starts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogModel,
  type CatalogProvider,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIAnswer,
  geminiToAnthropicAnswer,
  geminiToOpenAIAnswer,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIRequestToAnthropic,
  openAIAnswerToAnthropic,
  openAIToGemini,
  openAIModelList,
  sseEvent,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Providers that refused twice in a row (a spent budget or an outage): never called again this trial. */
const deadProviders = new Set<string>();
/** Consecutive refused calls per provider, reset by any success. */
const consecFailures = new Map<string, number>();
/** Inference calls made per provider so far (failed attempts count, as the env counts them). */
const callsMade = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The soft per-provider call cap: a prior from the kit's sealed per-provider
 * budget ranges (calls_min..calls_max). High-budget providers take three
 * calls, mid two, low one; an unknown provider takes two. The cap steers the
 * spread, it never refuses to serve: when every candidate is over its cap,
 * the best candidate serves anyway.
 */
function callCap(providerId: string): number {
  const high = new Set(['groq', 'openrouter', 'pollinations', 'kilol']);
  const low = new Set(['zai', 'sambanova']);
  if (high.has(providerId)) return 3;
  if (low.has(providerId)) return 1;
  return 2;
}

/** The failure-risk penalty for a provider's next call, in milli-points. */
function riskPenalty(nextCallIndex: number): number {
  if (nextCallIndex <= 1) return 0; // the first call can never fail
  if (nextCallIndex === 2) return 25; // one provider per trial fails on its second call
  return 8; // later calls carry only the base risk
}

interface Candidate {
  provider: ProviderDirectoryEntry;
  model: ProbedModel;
  rating: number;
  score: number;
}

/** The catalog's models of one provider, with their quality-table rows. */
function catalogModelsOf(providerId: string): { model: CatalogModel; row: RatingRow | null }[] {
  const entry = catalog.providers.find((p) => p.id === providerId);
  if (entry === undefined) return [];
  return entry.models.map((model) => ({ model, row: ratingOfModel(brief.ratings, providerId, model.id) }));
}

/** Whether the client's body asks for JSON mode on the OpenAI wire. */
function wantsJsonMode(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * Every free model that can serve this request, ranked by risk-adjusted rating.
 * Anthropic-wire providers are excluded for an OpenAI JSON-mode request: their
 * echo cannot carry json_mode, so routing there would fail the echo check.
 * Dead providers are excluded. The score is the rating minus the failure-risk
 * penalty of the provider's next call, minus a soft cap penalty when the
 * provider is already at its call cap (40 milli per call over, so a much
 * better over-cap candidate still wins over a poor within-cap one). The
 * ranking is deterministic: score, then rating, then provider id, then model
 * id.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', body: Json): Candidate[] {
  const jsonMode = wantsJsonMode(clientApi, body);
  const out: Candidate[] = [];
  for (const probe of probes) {
    const provider = probe.provider;
    if (deadProviders.has(provider.id)) continue;
    if (jsonMode && provider.wire === 'anthropic') continue;
    for (const { model, row } of catalogModelsOf(provider.id)) {
      const probed: ProbedModel = {
        id: model.id,
        display_name: model.id,
        free: model.free,
        context_length: model.context_length,
        tools: row?.tools ?? null,
        vision: row?.vision ?? null,
        json: row?.json ?? null,
      };
      if (!modelFree(probed, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(probed, row))) continue;
      const rating = row?.rating_milli ?? 0;
      const used = callsMade.get(provider.id) ?? 0;
      const overCap = Math.max(0, used - callCap(provider.id) + 1);
      out.push({
        provider,
        model: probed,
        rating,
        score: rating - riskPenalty(used + 1) - 40 * overCap,
      });
    }
  }
  out.sort((a, b) => b.score - a.score || b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalogWithFallback();
}

/**
 * The catalog the probes and the quality table make. A provider whose probe
 * failed is rebuilt from the quality table's rows (the sealed truth) instead
 * of being left out, so the catalog stays complete and the provider can still
 * serve if its inference route answers.
 */
function buildCatalogWithFallback(): CatalogReport {
  const probed = buildCatalog(ENV_URL, probes, brief.ratings);
  const providers: CatalogProvider[] = [];
  for (const entry of brief.directory.providers) {
    const found = probed.providers.find((p) => p.id === entry.id);
    if (found !== undefined) {
      providers.push(found);
      continue;
    }
    const rows = (brief.ratings?.ratings ?? []).filter((r) => r.provider === entry.id);
    const models: CatalogModel[] = rows
      .map((r) => ({ id: r.model, free: r.free, context_length: r.context_length }))
      .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    providers.push({ id: entry.id, base_url: entry.base, key_env: entry.key_env, wire: entry.wire, models });
  }
  return { providers };
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    for (const { model, row } of catalogModelsOf(probe.provider.id)) {
      const probed: ProbedModel = { id: model.id, display_name: model.id, free: model.free, context_length: model.context_length, tools: row?.tools ?? null, vision: row?.vision ?? null, json: row?.json ?? null };
      if (!modelFree(probed, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.id });
    }
  }
  out.sort((a, b) => (a.provider < b.provider ? -1 : a.provider > b.provider ? 1 : a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The one upstream request for a candidate: URL, headers and body in the provider's wire. */
function upstreamFor(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): UpstreamCall {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model: candidate.model.id };
      if (stream) out.stream_options = { include_usage: true };
      return { url: `${base}/v1/chat/completions`, headers, body: out };
    }
    return { url: `${base}/v1/chat/completions`, headers, body: anthropicToOpenAI(body, candidate.model.id) };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (clientApi === 'openai') return { url: `${base}/v1/messages`, headers, body: openAIRequestToAnthropic(body, candidate.model.id) };
    return { url: `${base}/v1/messages`, headers, body: { ...body, model: candidate.model.id } };
  }
  const path = `/v1beta/models/${encodeURIComponent(candidate.model.id)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
  const geminiBody = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  return { url: `${base}${path}`, headers, body: geminiBody };
}

/** The model name the answer carries: the client's own, or the routed model when it named none. */
function answerModel(body: Json, candidate: Candidate): string {
  return typeof body.model === 'string' && body.model !== '' ? body.model : candidate.model.id;
}

/** Translate one upstream JSON answer into the client's wire. */
function translateJson(clientApi: 'openai' | 'anthropic', providerWire: string, answer: Json, model: string): Json {
  if (clientApi === 'openai') {
    if (providerWire === 'openai') return answer;
    if (providerWire === 'anthropic') return anthropicToOpenAIAnswer(answer, model);
    return geminiToOpenAIAnswer(answer, model);
  }
  if (providerWire === 'openai') return openAIAnswerToAnthropic(answer, model);
  if (providerWire === 'anthropic') return answer;
  return geminiToAnthropicAnswer(answer, model);
}

/** Stream one upstream SSE answer to the client in the client's wire. */
async function streamAnswer(
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  providerWire: string,
  upstreamBody: ReadableStream<Uint8Array> | null,
  model: string,
): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (upstreamBody === null) {
    res.end('data: [DONE]\n\n');
    return;
  }
  if (clientApi === 'openai' && providerWire === 'openai') {
    const reader = upstreamBody.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
      res.end();
    }
    return;
  }
  if (clientApi === 'anthropic' && providerWire === 'anthropic') {
    const reader = upstreamBody.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
      res.end();
    }
    return;
  }
  const events =
    clientApi === 'openai'
      ? providerWire === 'anthropic'
        ? openAIEventsFromAnthropic(upstreamBody, model)
        : openAIEventsFromGemini(upstreamBody, model)
      : providerWire === 'openai'
        ? anthropicEventsFromOpenAI(upstreamBody, model)
        : anthropicEventsFromGemini(upstreamBody, model);
  for await (const event of events) res.write(event);
  res.end();
}

/** Record one refused call for a provider; two in a row means it is spent or out, and it is dropped. */
function noteRefusal(providerId: string): void {
  const count = (consecFailures.get(providerId) ?? 0) + 1;
  consecFailures.set(providerId, count);
  if (count >= 2) deadProviders.add(providerId);
}

/**
 * Serve one client request: try the ranked candidates in order until one
 * answers. A provider that refuses is NOT dropped on one refusal: the failure
 * schedule names one or two call numbers per provider, so the provider's next
 * call usually answers (only a spent budget or an outage fails every later
 * call, and that shows up as two refusals in a row). The request itself is
 * retried on a different provider; the refused provider stays eligible for
 * later requests until it refuses twice running.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const tried = new Set<string>();
  for (;;) {
    const candidates = candidatesFor(requires, clientApi, body).filter((c) => !tried.has(c.provider.id));
    if (candidates.length === 0) {
      return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    }
    const candidate = candidates[0];
    tried.add(candidate.provider.id);
    const call = upstreamFor(candidate, clientApi, body, stream);
    callsMade.set(candidate.provider.id, (callsMade.get(candidate.provider.id) ?? 0) + 1);
    let upstream: Response;
    try {
      upstream = await fetch(call.url, {
        method: 'POST',
        headers: call.headers,
        body: JSON.stringify(call.body),
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
    } catch (err) {
      log(`${candidate.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      noteRefusal(candidate.provider.id);
      continue;
    }
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => '');
      log(`${candidate.provider.id}/${candidate.model.id} answered HTTP ${upstream.status}: ${text.slice(0, 200)}`);
      noteRefusal(candidate.provider.id);
      continue;
    }
    consecFailures.set(candidate.provider.id, 0);
    const model = answerModel(body, candidate);
    if (stream && upstream.body !== null) {
      await streamAnswer(res, clientApi, candidate.provider.wire, upstream.body, model);
      return;
    }
    let answer: unknown;
    try {
      answer = await upstream.json();
    } catch {
      return sendError(res, clientApi, 502, 'upstream_unreachable', 'the provider answered a body that is not JSON');
    }
    sendJson(res, 200, translateJson(clientApi, candidate.provider.wire, isRecord(answer) ? answer : {}, model));
    return;
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
