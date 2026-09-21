/**
 * External contracts for the snapcompact vision-recall sidecar.
 */
import { afterEach, describe, expect, it, vi } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { Agent } from "@oh-my-pi/pi-agent-core";
import * as compactionModule from "@oh-my-pi/pi-agent-core/compaction";
import * as ai from "@oh-my-pi/pi-ai";
import type { AssistantMessage, Context, Message, Model } from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import {
	BUILTIN_TOOLS,
	HIDDEN_TOOLS,
	type Tool,
	type ToolSession,
	type XdevState,
} from "@oh-my-pi/pi-coding-agent/tools";
import { SnapcompactRecallTool } from "@oh-my-pi/pi-coding-agent/tools/snapcompact-recall";
import { TempDir } from "@oh-my-pi/pi-utils";
import * as snapcompact from "@oh-my-pi/snapcompact";
import { createAssistantMessage } from "./helpers/agent-session-setup";

const UNIQUE_FACT = "EACCES: permission denied on /usr/local/bin/node";

function assistantMessage(text: string, model: Model): Message {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		stopReason: "stop",
		usage: {
			input: 10,
			output: 10,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 20,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		timestamp: Date.now(),
	};
}

function sidecarAnswer(): AssistantMessage {
	return {
		role: "assistant",
		api: "anthropic-messages",
		provider: "aimlapi",
		model: "claude-sonnet-4-5-20250929",
		stopReason: "stop",
		content: [{ type: "text", text: `The deploy failed with ${UNIQUE_FACT}.` }],
		usage: {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		timestamp: Date.now(),
	} as unknown as AssistantMessage;
}

function imageDataIn(messages: readonly unknown[]): string[] {
	const found: string[] = [];
	const walk = (value: unknown): void => {
		if (Array.isArray(value)) {
			for (const item of value) walk(item);
			return;
		}
		if (!value || typeof value !== "object") return;
		if ("type" in value && value.type === "image" && "data" in value && typeof value.data === "string") {
			found.push(value.data);
			return;
		}
		for (const item of Object.values(value)) walk(item);
	};
	walk(messages);
	return found;
}

function allTextIn(messages: readonly unknown[]): string {
	const parts: string[] = [];
	const walk = (value: unknown): void => {
		if (Array.isArray(value)) {
			for (const item of value) walk(item);
			return;
		}
		if (!value || typeof value !== "object") return;
		if ("type" in value && value.type === "text" && "text" in value && typeof value.text === "string") {
			parts.push(value.text);
			return;
		}
		for (const item of Object.values(value)) walk(item);
	};
	walk(messages);
	return parts.join("\n");
}

interface Harness {
	session: AgentSession;
	sessionManager: SessionManager;
	activeModel: Model;
	settings: Settings;
	modelRegistry: ModelRegistry;
	toolSession: ToolSession;
}

describe("AgentSession snapcompact vision recall", () => {
	let session: AgentSession | undefined;
	const replaySessions: AgentSession[] = [];
	let authStorage: AuthStorage | undefined;
	let tempDir: TempDir | undefined;

	afterEach(async () => {
		try {
			await session?.dispose();
			for (const replaySession of replaySessions.splice(0)) await replaySession.dispose();
		} finally {
			authStorage?.close();
			await tempDir?.remove();
			vi.restoreAllMocks();
			ai.clearCustomApis();
			session = undefined;
			authStorage = undefined;
			tempDir = undefined;
		}
	});

	async function createHarness(
		options: {
			visionRole?: string | null;
			seedMessages?: Message[];
			failRecallTool?: boolean;
			xdevActive?: boolean;
		} = {},
	): Promise<Harness> {
		const activeModel = getBundledModel("aimlapi", "alibaba/qwen3-coder-480b-a35b-instruct");
		if (!activeModel) throw new Error("Expected bundled text-only model");
		expect(activeModel.input).not.toContain("image");

		tempDir = TempDir.createSync("@pi-snapcompact-recall-");
		authStorage = await AuthStorage.create(path.join(tempDir.path(), "auth.db"));
		authStorage.setRuntimeApiKey("aimlapi", "test-key");
		const modelRegistry = new ModelRegistry(authStorage);

		const sessionManager = SessionManager.create(tempDir.path(), tempDir.path());
		const seed = options.seedMessages ?? [
			{ role: "user", content: `I ran the deploy script. ${"alpha ".repeat(60000)}`, timestamp: Date.now() },
			{
				role: "assistant",
				content: [{ type: "text", text: `The deploy failed with error ${UNIQUE_FACT}. Try running with sudo.` }],
				api: activeModel.api,
				provider: activeModel.provider,
				model: activeModel.id,
				stopReason: "stop",
				usage: {
					input: 10,
					output: 10,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 20,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				timestamp: Date.now(),
			},
			{ role: "user", content: "second question", timestamp: Date.now() },
		];
		for (const message of seed) sessionManager.appendMessage(message);
		if (!sessionManager.getBranch()[0]?.id) throw new Error("Expected seeded branch entry");

		const settings = Settings.isolated({
			"compaction.methodOrder": ["snapcompact", "soft"],
			"compaction.keepRecentTokens": 1,
			...(options.xdevActive ? { "checkpoint.enabled": true } : {}),
			...(options.visionRole === null
				? {}
				: { modelRoles: { vision: options.visionRole ?? "aimlapi/claude-sonnet-4-5-20250929" } }),
		});

		const toolSession: ToolSession = {
			cwd: tempDir.path(),
			hasUI: false,
			getSessionFile: () => null,
			getSessionSpawns: () => "*",
			settings,
			modelRegistry,
			sessionManager,
			getActiveModel: () => sessionRef.model,
			getSessionId: () => sessionRef.sessionId,
			getTelemetry: () => undefined,
		} as unknown as ToolSession;

		// Optional xdev presentation layer: real built-in tools in the canonical
		// registry so discoverable tools (checkpoint/rewind) mount under xd:// and
		// only pinned tools stay top-level.
		let agentTools: Tool[] = [];
		let sessionExtras: {
			toolRegistry?: Map<string, Tool>;
			builtInToolNames?: string[];
			xdev?: XdevState;
		} = {};
		if (options.xdevActive) {
			const names = ["read", "write", "checkpoint", "rewind"] as const;
			const toolRegistry = new Map<string, Tool>();
			for (const name of names) {
				const tool = await BUILTIN_TOOLS[name](toolSession);
				if (!tool) throw new Error(`Expected built-in tool ${name}`);
				toolRegistry.set(name, tool);
			}
			agentTools = [...toolRegistry.values()];
			sessionExtras = {
				toolRegistry,
				builtInToolNames: [...names],
				xdev: {
					tools: toolRegistry,
					mountedNames: new Set<string>(),
					builtInNames: new Set(names),
					isActive: () => false,
				},
			};
		}

		const agent = new Agent({
			initialState: { model: activeModel, systemPrompt: ["Test"], tools: agentTools, messages: [] },
		});
		const sessionRef: AgentSession = new AgentSession({
			agent,
			sessionManager,
			settings,
			modelRegistry,
			createSnapcompactRecallTool: options.failRecallTool
				? async () => null
				: async () => (await HIDDEN_TOOLS.snapcompact_recall(toolSession)) ?? null,
			...sessionExtras,
		});

		return { session: sessionRef, sessionManager, activeModel, settings, modelRegistry, toolSession };
	}

	function activeToolNames(harness: Harness): string[] {
		return harness.session.agent.state.tools.map(tool => tool.name);
	}

	function makeToolSession(
		harness: Harness,
		overrides: { settings?: Settings; modelRegistry?: ModelRegistry },
	): ToolSession {
		return {
			cwd: tempDir?.path() ?? ".",
			hasUI: false,
			getSessionFile: () => null,
			getSessionSpawns: () => "*",
			settings: overrides.settings ?? harness.settings,
			modelRegistry: overrides.modelRegistry ?? harness.modelRegistry,
			sessionManager: harness.sessionManager,
			getActiveModel: () => harness.session.model,
			getSessionId: () => harness.session.sessionId,
			getTelemetry: () => undefined,
		} as unknown as ToolSession;
	}

	it("rebuilds the agent context image-free with a recall-availability marker", async () => {
		const harness = await createHarness({
			seedMessages: [
				{ role: "user", content: `I ran the deploy script. ${"alpha ".repeat(30000)}`, timestamp: Date.now() },
				{
					role: "assistant",
					content: [{ type: "text", text: `The deploy failed with error ${UNIQUE_FACT}. Try running with sudo.` }],
					api: getBundledModel("aimlapi", "alibaba/qwen3-coder-480b-a35b-instruct")!.api,
					provider: getBundledModel("aimlapi", "alibaba/qwen3-coder-480b-a35b-instruct")!.provider,
					model: getBundledModel("aimlapi", "alibaba/qwen3-coder-480b-a35b-instruct")!.id,
					stopReason: "stop",
					usage: {
						input: 10,
						output: 10,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 20,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					timestamp: Date.now(),
				},
				{ role: "user", content: "second question", timestamp: Date.now() },
			],
		});
		await harness.session.compact(undefined, { mode: "snapcompact" });

		const compaction = harness.sessionManager.getBranch().find(entry => entry.type === "compaction");
		const archive = snapcompact.getPreservedArchive(compaction!.preserveData);
		expect(archive?.frames.length ?? 0).toBeGreaterThan(0);

		const context = harness.sessionManager.buildSessionContext();
		expect(imageDataIn(context.messages)).toEqual([]);
		const text = allTextIn(context.messages);
		expect(text).toContain("I ran the deploy script");
		expect(text.toLowerCase()).toContain("snapcompact_recall");
	});

	it("routes a focused recall query to the vision role and returns only its text", async () => {
		const harness = await createHarness();
		await harness.session.compact(undefined, { mode: "snapcompact" });
		expect(activeToolNames(harness)).toContain("snapcompact_recall");

		const modelBefore = harness.session.model;
		const completeSpy = vi.spyOn(ai, "completeSimple").mockResolvedValue({
			role: "assistant",
			api: "anthropic-messages",
			provider: "aimlapi",
			model: "claude-sonnet-4-5-20250929",
			stopReason: "stop",
			content: [{ type: "text", text: `The deploy failed with ${UNIQUE_FACT}.` }],
			usage: {
				input: 1,
				output: 1,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 2,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			timestamp: Date.now(),
		} as unknown as AssistantMessage);

		const tool = new SnapcompactRecallTool(harness.toolSession);
		const result = await tool.execute("call-1", { query: "What error did the deploy fail with?" });

		expect(completeSpy).toHaveBeenCalledTimes(1);
		const [model, ctx] = completeSpy.mock.calls[0] as [Model, ai.Context];
		expect(`${model.provider}/${model.id}`).toBe("aimlapi/claude-sonnet-4-5-20250929");
		expect(`${model.provider}/${model.id}`).not.toBe(`${harness.activeModel.provider}/${harness.activeModel.id}`);
		expect(ctx.tools).toBeUndefined();
		expect(ctx.messages).toHaveLength(1);
		const userMessage = ctx.messages[0];
		expect(userMessage.role).toBe("user");
		type Block = { type: string; text?: string; data?: string; mimeType?: string };
		const content = userMessage.content as Block[];
		const compaction = harness.sessionManager.getBranch().find(entry => entry.type === "compaction");
		expect(content[0].type).toBe("text");
		expect(content[0].text).toBe(compaction!.summary);
		const recallContext = harness.sessionManager.buildSessionContext({ includeSnapcompactFrames: true });
		const summaryMessage = recallContext.messages.find(message => message.role === "compactionSummary");
		expect(summaryMessage).toBeDefined();
		const expectedBlocks = summaryMessage && summaryMessage.role === "compactionSummary" ? summaryMessage.blocks : [];
		if (!expectedBlocks) throw new Error("Expected blocks not found");
		expect(content.slice(1, -1)).toEqual(expectedBlocks);
		expect(content.at(-1)?.type).toBe("text");
		expect(content.at(-1)?.text).toContain("What error did the deploy fail with?");
		const archive = snapcompact.getPreservedArchive(compaction!.preserveData);
		const imageBlocks = content.filter(block => block.type === "image");
		expect(imageBlocks.length).toBe(archive?.frames.length ?? 0);
		expect(result.content.map(block => (block.type === "text" ? block.text : "")).join("")).toContain(UNIQUE_FACT);
		expect(harness.session.model).toBe(modelBefore);
	});

	it("keeps the active model unchanged across a recall execution", async () => {
		const harness = await createHarness();
		await harness.session.compact(undefined, { mode: "snapcompact" });
		const modelBefore = harness.session.model;
		vi.spyOn(ai, "completeSimple").mockResolvedValue({
			role: "assistant",
			api: "anthropic-messages",
			provider: "aimlapi",
			model: "claude-sonnet-4-5-20250929",
			stopReason: "stop",
			content: [{ type: "text", text: `The deploy failed with ${UNIQUE_FACT}.` }],
			usage: {
				input: 1,
				output: 1,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 2,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			timestamp: Date.now(),
		} as unknown as AssistantMessage);

		const tool = new SnapcompactRecallTool(harness.toolSession);
		await tool.execute("call-1", { query: "What error did the deploy fail with?" });

		expect(harness.session.model).toBe(modelBefore);
		expect(harness.session.model?.id).toBe(harness.activeModel.id);
	});

	it("is inactive before a frame-bearing snapcompact and after a superseding compaction", async () => {
		const harness = await createHarness({
			visionRole: "aimlapi/claude-sonnet-4-5-20250929",
			seedMessages: [
				{ role: "user", content: `I ran the deploy script. ${"alpha ".repeat(60000)}`, timestamp: Date.now() },
				{
					role: "assistant",
					content: [{ type: "text", text: `The deploy failed with error ${UNIQUE_FACT}. Try running with sudo.` }],
					api: getBundledModel("aimlapi", "alibaba/qwen3-coder-480b-a35b-instruct")!.api,
					provider: getBundledModel("aimlapi", "alibaba/qwen3-coder-480b-a35b-instruct")!.provider,
					model: getBundledModel("aimlapi", "alibaba/qwen3-coder-480b-a35b-instruct")!.id,
					stopReason: "stop",
					usage: {
						input: 10,
						output: 10,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 20,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					timestamp: Date.now(),
				},
				{ role: "user", content: "second question", timestamp: Date.now() },
			],
		});
		expect(activeToolNames(harness)).not.toContain("snapcompact_recall");

		await harness.session.compact(undefined, { mode: "snapcompact" });
		expect(activeToolNames(harness)).toContain("snapcompact_recall");

		vi.spyOn(compactionModule, "compact").mockImplementation(async (preparation, model) => ({
			summary: "soft summary",
			shortSummary: "soft",
			firstKeptEntryId: preparation.firstKeptEntryId,
			tokensBefore: 42,
			details: { provider: model.provider, model: model.id },
		}));

		await harness.sessionManager.appendMessage({ role: "user", content: "third question", timestamp: Date.now() });
		await harness.sessionManager.appendMessage(assistantMessage("third answer", harness.session.model!));
		await harness.sessionManager.appendMessage({ role: "user", content: "fourth question", timestamp: Date.now() });
		await harness.sessionManager.appendMessage(assistantMessage("fourth answer", harness.session.model!));

		await harness.session.compact(undefined, { mode: "soft" });
		expect(activeToolNames(harness)).not.toContain("snapcompact_recall");
	});

	it("deactivates after navigating to a branch without the archive", async () => {
		const harness = await createHarness();
		const firstEntryId = harness.sessionManager.getBranch()[0]?.id;
		if (!firstEntryId) throw new Error("Expected a first branch entry");
		await harness.session.compact(undefined, { mode: "snapcompact" });
		expect(activeToolNames(harness)).toContain("snapcompact_recall");

		const result = await harness.session.navigateTree(firstEntryId);
		expect(result.cancelled).toBe(false);
		expect(activeToolNames(harness)).not.toContain("snapcompact_recall");
	});

	it("surfaces a warning notice once per archive when the recall tool cannot be enabled", async () => {
		const harness = await createHarness({ failRecallTool: true });
		const firstEntryId = harness.sessionManager.getBranch()[0]?.id;
		if (!firstEntryId) throw new Error("Expected a first branch entry");
		const notices: { level: string; message: string; source?: string }[] = [];
		harness.session.subscribe(event => {
			if (event.type === "notice") {
				notices.push({ level: event.level, message: event.message, source: event.source });
			}
		});

		await harness.session.compact(undefined, { mode: "snapcompact" });
		expect(activeToolNames(harness)).not.toContain("snapcompact_recall");
		const leafId = harness.sessionManager.getBranch().at(-1)?.id;
		if (!leafId) throw new Error("Expected a branch leaf");

		// Re-syncing the same archive through navigation must not repeat the warning.
		const away = await harness.session.navigateTree(firstEntryId);
		expect(away.cancelled).toBe(false);
		const back = await harness.session.navigateTree(leafId);
		expect(back.cancelled).toBe(false);

		const recallNotices = notices.filter(
			event => event.source === "compaction" && event.message.includes("snapcompact_recall"),
		);
		expect(recallNotices).toHaveLength(1);
		expect(recallNotices[0].level).toBe("warning");
	});

	it("exposes a frame-less archive as plain text without activating the tool", async () => {
		const activeModel = getBundledModel("aimlapi", "alibaba/qwen3-coder-480b-a35b-instruct");
		if (!activeModel) throw new Error("Expected bundled text-only model");
		const harness = await createHarness({
			visionRole: "aimlapi/claude-sonnet-4-5-20250929",
			seedMessages: [
				{ role: "user", content: `first question ${"alpha ".repeat(4000)}`, timestamp: Date.now() },
				{
					role: "assistant",
					content: [{ type: "text", text: "tiny answer" }],
					api: activeModel.api,
					provider: activeModel.provider,
					model: activeModel.id,
					stopReason: "stop",
					usage: {
						input: 10,
						output: 10,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 20,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					timestamp: Date.now(),
				},
				{ role: "user", content: "second question", timestamp: Date.now() },
			],
		});
		await harness.session.compact(undefined, { mode: "snapcompact" });
		const compaction = harness.sessionManager.getBranch().find(entry => entry.type === "compaction");
		expect(compaction).toBeDefined();
		const archive = snapcompact.getPreservedArchive(compaction!.preserveData);
		expect(archive?.frames.length ?? 0).toBe(0);
		const text = allTextIn(harness.sessionManager.buildSessionContext().messages);
		expect(text).toContain("tiny answer");
		expect(activeToolNames(harness)).not.toContain("snapcompact_recall");
		const tool = new SnapcompactRecallTool(harness.toolSession);
		const completeSpy = vi.spyOn(ai, "completeSimple");
		await expect(
			tool.execute("call-no-frames", { query: "What was the answer to the first question?" }),
		).rejects.toThrow(/image-bearing/i);
		expect(completeSpy).not.toHaveBeenCalled();
	});

	it("fails with an actionable ToolError and no fabricated answer on sidecar failure", async () => {
		const harness = await createHarness();
		await harness.session.compact(undefined, { mode: "snapcompact" });
		const tool = new SnapcompactRecallTool(harness.toolSession);

		vi.spyOn(ai, "completeSimple").mockResolvedValue({
			role: "assistant",
			api: "anthropic-messages",
			provider: "aimlapi",
			model: "claude-sonnet-4-5-20250929",
			stopReason: "error",
			errorMessage: "provider exploded",
			content: [],
			usage: {
				input: 1,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 1,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			timestamp: Date.now(),
		} as unknown as AssistantMessage);
		await expect(tool.execute("call-err", { query: "What error did the deploy fail with?" })).rejects.toThrow(
			/provider exploded/,
		);

		vi.spyOn(ai, "completeSimple").mockResolvedValue({
			role: "assistant",
			api: "anthropic-messages",
			provider: "aimlapi",
			model: "claude-sonnet-4-5-20250929",
			stopReason: "aborted",
			content: [],
			usage: {
				input: 1,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 1,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			timestamp: Date.now(),
		} as unknown as AssistantMessage);
		await expect(tool.execute("call-abort", { query: "What error did the deploy fail with?" })).rejects.toThrow(
			/abort/i,
		);

		vi.spyOn(ai, "completeSimple").mockResolvedValue({
			role: "assistant",
			api: "anthropic-messages",
			provider: "aimlapi",
			model: "claude-sonnet-4-5-20250929",
			stopReason: "stop",
			content: [],
			usage: {
				input: 1,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 1,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			timestamp: Date.now(),
		} as unknown as AssistantMessage);
		await expect(tool.execute("call-empty", { query: "What error did the deploy fail with?" })).rejects.toThrow(
			/no text/i,
		);
	});

	it("rejects an empty query before any sidecar request", async () => {
		const harness = await createHarness();
		await harness.session.compact(undefined, { mode: "snapcompact" });
		const tool = new SnapcompactRecallTool(harness.toolSession);
		const completeSpy = vi.spyOn(ai, "completeSimple");
		await expect(tool.execute("call-empty-query", { query: "   " })).rejects.toThrow(/query/i);
		expect(completeSpy).not.toHaveBeenCalled();
	});

	it("fails with an actionable ToolError when the vision role is missing or unauthenticated", async () => {
		const harness = await createHarness();
		await harness.session.compact(undefined, { mode: "snapcompact" });
		const compaction = harness.sessionManager.getBranch().find(entry => entry.type === "compaction");
		const archive = snapcompact.getPreservedArchive(compaction!.preserveData);
		expect(archive?.frames.length ?? 0).toBeGreaterThan(0);

		const noRoleSettings = Settings.isolated({
			"compaction.methodOrder": ["snapcompact", "soft"],
			"compaction.keepRecentTokens": 1,
		});
		const noRoleTool = new SnapcompactRecallTool(makeToolSession(harness, { settings: noRoleSettings }));
		const completeSpy = vi.spyOn(ai, "completeSimple");
		await expect(
			noRoleTool.execute("call-no-vision", { query: "What error did the deploy fail with?" }),
		).rejects.toThrow(/vision/i);
		expect(completeSpy).not.toHaveBeenCalled();

		const keylessAuth = await AuthStorage.create(":memory:");
		try {
			const keylessRegistry = new ModelRegistry(keylessAuth);
			const noCredsTool = new SnapcompactRecallTool(makeToolSession(harness, { modelRegistry: keylessRegistry }));
			await expect(
				noCredsTool.execute("call-no-creds", { query: "What error did the deploy fail with?" }),
			).rejects.toThrow(/vision/i);
			expect(completeSpy).not.toHaveBeenCalled();
		} finally {
			keylessAuth.close();
		}
	});

	it("keeps snapcompact_recall top-level while xdev mounts other discoverable tools", async () => {
		const harness = await createHarness({ xdevActive: true });
		await harness.session.compact(undefined, { mode: "snapcompact" });

		expect(activeToolNames(harness)).toContain("snapcompact_recall");
		const mounted = harness.session.getMountedXdevToolNames();
		// The pin only matters if xdev is actually mounting other discoverables.
		expect(mounted).toContain("checkpoint");
		expect(mounted).not.toContain("snapcompact_recall");
	});

	it("persists recall calls under the canonical name across events, history, and replay", async () => {
		const harness = await createHarness();
		await harness.session.compact(undefined, { mode: "snapcompact" });
		vi.spyOn(ai, "completeSimple").mockResolvedValue(sidecarAnswer());

		const callId = "call-recall-canonical";
		const api = "test-snapcompact-recall-turn";
		let requests = 0;
		const contexts: Context[] = [];
		ai.registerCustomApi(api, (_model, context) => {
			requests += 1;
			contexts.push(context);
			const stream = new AssistantMessageEventStream();
			queueMicrotask(() => {
				if (requests === 1) {
					const message = createAssistantMessage("");
					const toolCall = {
						type: "toolCall",
						id: callId,
						name: "snapcompact_recall",
						arguments: { query: "What error did the deploy fail with?" },
					} as const;
					message.content = [toolCall];
					message.stopReason = "toolUse";
					stream.push({ type: "start", partial: message });
					stream.push({ type: "toolcall_start", contentIndex: 0, partial: message });
					stream.push({ type: "toolcall_end", contentIndex: 0, toolCall: toolCall as never, partial: message });
					stream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage("done");
					stream.push({ type: "start", partial: message });
					stream.push({ type: "done", reason: "stop", message });
				}
			});
			return stream;
		});
		harness.session.agent.setModel({ ...harness.activeModel, api: api as never });

		const toolEvents: string[] = [];
		harness.session.subscribe(event => {
			if (event.type === "tool_execution_start" || event.type === "tool_execution_end") {
				toolEvents.push(`${event.type}:${event.toolName}`);
			}
		});

		await harness.session.sendUserMessage("what failed?");

		expect(toolEvents).toEqual(["tool_execution_start:snapcompact_recall", "tool_execution_end:snapcompact_recall"]);

		const branch = harness.sessionManager.getBranch();
		const callEntry = branch.find(
			entry =>
				entry.type === "message" &&
				entry.message.role === "assistant" &&
				entry.message.content.some(block => block.type === "toolCall" && block.id === callId),
		);
		expect(callEntry).toBeDefined();
		const resultEntry = branch.find(
			entry =>
				entry.type === "message" && entry.message.role === "toolResult" && entry.message.toolCallId === callId,
		);
		if (!resultEntry || resultEntry.type !== "message" || resultEntry.message.role !== "toolResult") {
			throw new Error("Expected a persisted recall tool result");
		}
		expect(resultEntry.message.toolName).toBe("snapcompact_recall");

		const jsonlName = fs.readdirSync(tempDir!.path()).find(name => name.endsWith(".jsonl"));
		if (!jsonlName) throw new Error("Expected a persisted session file");
		const lines = fs.readFileSync(path.join(tempDir!.path(), jsonlName), "utf8").split("\n").filter(Boolean);
		const callLines = lines.filter(line => line.includes(callId));
		expect(callLines.length).toBeGreaterThan(0);
		for (const line of callLines) {
			expect(line).toContain("snapcompact_recall");
			expect(line).not.toContain("xd://");
		}

		// Reopen the persisted session and verify the recall exchange replays as a
		// paired canonical-name pair — never as a write envelope to an xd:// device.
		const jsonlPath = path.join(tempDir!.path(), jsonlName);
		const reopened = await SessionManager.open(jsonlPath, tempDir!.path(), undefined, {
			initialCwd: tempDir!.path(),
		});
		let replaySessionRef: AgentSession;
		const replayToolSession: ToolSession = {
			cwd: tempDir!.path(),
			hasUI: false,
			getSessionFile: () => null,
			getSessionSpawns: () => "*",
			settings: harness.settings,
			modelRegistry: harness.modelRegistry,
			sessionManager: reopened,
			getActiveModel: () => replaySessionRef.model,
			getSessionId: () => replaySessionRef.sessionId,
			getTelemetry: () => undefined,
		} as unknown as ToolSession;
		const replayAgent = new Agent({
			initialState: { model: harness.activeModel, systemPrompt: ["Test"], tools: [], messages: [] },
		});
		const replaySession = new AgentSession({
			agent: replayAgent,
			sessionManager: reopened,
			settings: harness.settings,
			modelRegistry: harness.modelRegistry,
			createSnapcompactRecallTool: async () => (await HIDDEN_TOOLS.snapcompact_recall(replayToolSession)) ?? null,
		});
		replaySessionRef = replaySession;
		replaySessions.push(replaySession);
		// Load the persisted conversation into the agent — the same seam switchSession
		// uses when resuming (buildDisplaySessionContext + replaceMessages).
		replaySession.agent.replaceMessages(replaySession.buildDisplaySessionContext().messages);
		replaySession.agent.setModel({ ...harness.activeModel, api: api as never });

		await replaySession.sendUserMessage("what failed?");

		const replayContext = contexts.at(-1);
		if (!replayContext) throw new Error("Expected a replay provider request");
		const replayCall = replayContext.messages.find(
			message =>
				message.role === "assistant" &&
				message.content.some(block => block.type === "toolCall" && block.id === callId),
		);
		expect(replayCall).toBeDefined();
		const replayResult = replayContext.messages.find(
			message => message.role === "toolResult" && message.toolCallId === callId,
		);
		if (!replayResult || replayResult.role !== "toolResult")
			throw new Error("Expected a replayed recall tool result");
		expect(replayResult.toolName).toBe("snapcompact_recall");
		expect(allTextIn([replayResult])).toContain(UNIQUE_FACT);
	}, 60_000);
});
