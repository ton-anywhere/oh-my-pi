import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "bun:test";
import { Agent, type AgentTool } from "@oh-my-pi/pi-agent-core";
import * as compactionModule from "@oh-my-pi/pi-agent-core/compaction";
import type { Message } from "@oh-my-pi/pi-ai";
import { type } from "@oh-my-pi/omptype";
import { type GeneratedProvider, getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import type { CompactionMethod } from "@oh-my-pi/pi-coding-agent/session/compaction-methods";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import * as snapcompact from "@oh-my-pi/snapcompact";

const UNRENDERABLE_SNAPCOMPACT_TEXT = "\uE000\uE001\uE002\uE003\uE004\uE005\uE006\uE007\uE008\uE009";

/** Minimal stand-in for the sidecar tool: the fallback tests only observe lifecycle activation. */
function makeRecallStub(): AgentTool {
	return {
		name: "snapcompact_recall",
		approval: "read" as const,
		label: "Snapcompact Recall",
		description: "Recall exact detail from the archived snapcompact history",
		parameters: type({ query: "string" }),
		async execute() {
			return { content: [{ type: "text" as const, text: "stub-recall" }] };
		},
	};
}

interface Harness {
	session: AgentSession;
	sessionManager: SessionManager;
	notices: string[];
	awaitCompactionEnd: () => Promise<{ action: string; errorMessage?: string }>;
	triggerThreshold: () => void;
}

interface HarnessOptions {
	activeModel: { provider: GeneratedProvider; id: string };
	seedMessages?: Message[];
	/** Null leaves compaction.methodOrder at its schema default. */
	methodOrder?: readonly CompactionMethod[] | null;
	/** Null omits modelRoles.vision, leaving a text-only active model with no reader. */
	visionRole?: string | null;
}

async function createHarness(modelRegistry: ModelRegistry, options: HarnessOptions): Promise<Harness> {
	const activeModel = getBundledModel(options.activeModel.provider, options.activeModel.id);
	if (!activeModel) throw new Error(`Missing bundled model ${options.activeModel.provider}/${options.activeModel.id}`);
	const agent = new Agent({
		initialState: { model: activeModel, systemPrompt: ["Test"], tools: [], messages: [] },
	});
	const sessionManager = SessionManager.inMemory();
	const seed = options.seedMessages ?? [{ role: "user", content: "hello", timestamp: Date.now() }];
	for (const message of seed) sessionManager.appendMessage(message);
	const firstKeptEntryId = sessionManager.getBranch()[0]?.id;
	if (!firstKeptEntryId) throw new Error("Expected seeded branch entry");

	const methodOrder = options.methodOrder ?? ["snapcompact", "soft"];
	const settings = Settings.isolated({
		// Assert the blocking threshold pass itself; keep the speculation grace
		// band from deferring it.
		"compaction.asyncEnabled": false,
		...(options.methodOrder === null ? {} : { "compaction.methodOrder": [...methodOrder] }),
		// Force a 1-token recent window so the post-turn cut always splits off the
		// last turn and summarizes the seeded unrenderable history. With the default
		// 20k window the cut keeps both tiny messages, leaving nothing for
		// snapcompact's renderability preflight to scan.
		"compaction.keepRecentTokens": 1,
		...(options.visionRole === null
			? {}
			: { modelRoles: { vision: options.visionRole ?? "aimlapi/claude-sonnet-4-5-20250929" } }),
	});
	const session = new AgentSession({
		agent,
		sessionManager,
		settings,
		modelRegistry,
		createSnapcompactRecallTool: async () => makeRecallStub(),
	});
	vi.spyOn(compactionModule, "compact").mockResolvedValue({
		summary: "compacted",
		shortSummary: undefined,
		firstKeptEntryId,
		tokensBefore: 123,
		details: {},
	});
	const end = Promise.withResolvers<{ action: string; errorMessage?: string }>();
	const notices: string[] = [];
	session.subscribe(event => {
		if (event.type === "notice" && event.source === "compaction") notices.push(event.message);
		if (
			event.type === "auto_compaction_end" &&
			!event.aborted &&
			(event.result !== undefined || event.skipped === true)
		) {
			end.resolve({ action: event.action, errorMessage: event.errorMessage });
		}
	});

	const triggerThreshold = () => {
		// Prompt tokens above the auto-compaction threshold but below the model's
		// context window: post-turn maintenance must run a threshold compaction,
		// NOT the overflow recovery path (which drops the just-ended turn before
		// snapcompact's renderability preflight can scan it, leaving nothing to
		// summarize). Derived from the live window so the fixture survives model
		// metadata changes (claude-sonnet-4-5's 200k window is narrower than the
		// vision-role qwen's, so a fixed count would overflow one of them).
		const contextWindow = activeModel.contextWindow ?? 0;
		const thresholdTokens = compactionModule.resolveThresholdTokens(contextWindow, settings.getGroup("compaction"));
		const promptTokens = contextWindow > 0 ? Math.floor((thresholdTokens + contextWindow) / 2) : 246_000;
		const assistantMsg = {
			role: "assistant" as const,
			content: [{ type: "text" as const, text: "Done." }],
			api: activeModel.api,
			provider: activeModel.provider,
			model: activeModel.id,
			stopReason: "stop" as const,
			usage: {
				input: promptTokens,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: promptTokens,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			timestamp: Date.now(),
		};
		session.agent.emitExternalEvent({ type: "message_end", message: assistantMsg });
		session.agent.emitExternalEvent({ type: "agent_end", messages: [assistantMsg] });
	};

	return { session, sessionManager, notices, awaitCompactionEnd: () => end.promise, triggerThreshold };
}

describe("AgentSession auto-snapcompact local-blocker fallback", () => {
	let session: AgentSession | undefined;
	let authStorage: AuthStorage;
	let modelRegistry: ModelRegistry;

	beforeAll(async () => {
		authStorage = await AuthStorage.create(":memory:");
		authStorage.setRuntimeApiKey("aimlapi", "test-key");
		authStorage.setRuntimeApiKey("openai", "test-key");
		modelRegistry = new ModelRegistry(authStorage);
	});

	afterEach(async () => {
		await session?.dispose();
		vi.restoreAllMocks();
		session = undefined;
	});

	afterAll(() => {
		authStorage.close();
	});

	it("uses snapcompact with the vision reader when the active model is text-only", async () => {
		const harness = await createHarness(modelRegistry, {
			activeModel: { provider: "aimlapi", id: "alibaba/qwen3-coder-480b-a35b-instruct" },
		});
		session = harness.session;
		harness.triggerThreshold();

		const result = await harness.awaitCompactionEnd();
		// A text-only active model is no longer a local blocker when
		// `modelRoles.vision` names a usable reader: the pass runs against the
		// vision reader while the session keeps the active conversation model.
		expect(result).toEqual({ action: "snapcompact", errorMessage: undefined });
		expect(compactionModule.compact).not.toHaveBeenCalled();
		expect(harness.sessionManager.getBranch().some(entry => entry.type === "compaction")).toBe(true);
		expect(harness.session.model?.id).toBe("alibaba/qwen3-coder-480b-a35b-instruct");
		expect(
			harness.sessionManager.getBranch().some(entry => entry.type === "model_change" && entry.role === "vision"),
		).toBe(false);
	});

	it("keeps the active model and activates recall when the vision reader archives frames", async () => {
		const harness = await createHarness(modelRegistry, {
			activeModel: { provider: "aimlapi", id: "alibaba/qwen3-coder-480b-a35b-instruct" },
			seedMessages: [
				{ role: "user", content: `first question ${"alpha ".repeat(30000)}`, timestamp: Date.now() },
				{ role: "user", content: "second question", timestamp: Date.now() },
			],
		});
		session = harness.session;
		harness.triggerThreshold();
		await harness.awaitCompactionEnd();

		const compaction = harness.sessionManager.getBranch().find(entry => entry.type === "compaction");
		expect(compaction).toBeDefined();
		// The committed archive carries frames, so the sidecar recall tool must
		// become active for the current branch.
		const archive = snapcompact.getPreservedArchive(compaction!.preserveData);
		expect(archive?.frames.length ?? 0).toBeGreaterThan(0);
		// The conversation model never switches: the archive is read by the
		// vision sidecar, not the active model.
		expect(harness.session.model?.id).toBe("alibaba/qwen3-coder-480b-a35b-instruct");
		expect(
			harness.sessionManager.getBranch().some(entry => entry.type === "model_change" && entry.role === "vision"),
		).toBe(false);
		const activeToolNames = harness.session.agent.state.tools.map(tool => tool.name);
		expect(activeToolNames).toContain("snapcompact_recall");
	});

	it("keeps the active model when a local blocker stops the vision reader", async () => {
		const harness = await createHarness(modelRegistry, {
			activeModel: { provider: "aimlapi", id: "alibaba/qwen3-coder-480b-a35b-instruct" },
			seedMessages: [
				{
					role: "user",
					content: UNRENDERABLE_SNAPCOMPACT_TEXT.repeat(10),
					timestamp: Date.now(),
				},
			],
		});
		session = harness.session;
		harness.triggerThreshold();

		const result = await harness.awaitCompactionEnd();
		// The glyph blocker rejects the pass before any render lands, so the
		// session must stay exactly as unswitched as if no role were configured.
		expect(result.action).toBe("context-full");
		expect(compactionModule.compact).toHaveBeenCalled();
		expect(harness.session.model?.id).toBe("alibaba/qwen3-coder-480b-a35b-instruct");
		expect(harness.sessionManager.getBranch().some(entry => entry.type === "model_change")).toBe(false);
	});

	it("uses soft compaction when snapcompact is unavailable for the active model", async () => {
		const harness = await createHarness(modelRegistry, {
			activeModel: { provider: "aimlapi", id: "alibaba/qwen3-coder-480b-a35b-instruct" },
			visionRole: null,
		});
		session = harness.session;
		harness.triggerThreshold();

		const result = await harness.awaitCompactionEnd();
		expect(result).toEqual({ action: "context-full", errorMessage: undefined });
		expect(compactionModule.compact).toHaveBeenCalled();
		expect(harness.sessionManager.getBranch().find(entry => entry.type === "compaction")).toMatchObject({
			type: "compaction",
			summary: "compacted",
		});
	});

	it("uses snapcompact for a non-OpenAI vision model under the default preference order", async () => {
		const harness = await createHarness(modelRegistry, {
			activeModel: { provider: "aimlapi", id: "claude-sonnet-4-5-20250929" },
			methodOrder: null,
		});
		session = harness.session;
		harness.triggerThreshold();

		const result = await harness.awaitCompactionEnd();

		expect(result).toEqual({ action: "snapcompact", errorMessage: undefined });
		expect(compactionModule.compact).not.toHaveBeenCalled();
		expect(harness.sessionManager.getBranch().some(entry => entry.type === "compaction")).toBe(true);
	});

	it("uses OpenAI server compaction before local fallback methods by default", async () => {
		const harness = await createHarness(modelRegistry, {
			activeModel: { provider: "openai", id: "gpt-5" },
			methodOrder: null,
		});
		session = harness.session;
		harness.triggerThreshold();

		const result = await harness.awaitCompactionEnd();

		expect(result).toEqual({ action: "remote", errorMessage: undefined });
		expect(compactionModule.compact).toHaveBeenCalledTimes(1);
	});

	it("falls through from a failed OpenAI server compaction to snapcompact", async () => {
		const harness = await createHarness(modelRegistry, {
			activeModel: { provider: "openai", id: "gpt-5" },
			methodOrder: null,
		});
		session = harness.session;
		vi.spyOn(compactionModule, "compact").mockRejectedValue(new Error("server compaction unavailable"));
		harness.triggerThreshold();

		const result = await harness.awaitCompactionEnd();

		expect(result).toEqual({ action: "snapcompact", errorMessage: undefined });
		expect(compactionModule.compact).toHaveBeenCalledTimes(1);
	});
	it("downgrades to context-full when unsupported glyphs make snapcompact unsafe", async () => {
		const harness = await createHarness(modelRegistry, {
			activeModel: { provider: "aimlapi", id: "claude-sonnet-4-5-20250929" },
			seedMessages: [
				{
					role: "user",
					content: UNRENDERABLE_SNAPCOMPACT_TEXT.repeat(10),
					timestamp: Date.now(),
				},
			],
		});
		session = harness.session;
		harness.triggerThreshold();

		const result = await harness.awaitCompactionEnd();
		expect(result.action).toBe("context-full");
		expect(result.errorMessage).toBeUndefined();
		expect(compactionModule.compact).toHaveBeenCalled();
		const unsupportedGlyphNotice = harness.notices.find(message =>
			message.startsWith("snapcompact disabled: unsupported characters for selected snapcompact font"),
		);
		expect(unsupportedGlyphNotice).toContain("trying the next preferred compaction method.");
		expect(harness.sessionManager.getBranch().find(entry => entry.type === "compaction")).toMatchObject({
			type: "compaction",
			summary: "compacted",
		});
	});
});
