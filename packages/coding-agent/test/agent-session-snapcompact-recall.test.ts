/**
 * External contracts for the snapcompact vision-recall sidecar.
 */
import { afterEach, describe, expect, it, vi } from "bun:test";
import * as path from "node:path";
import { Agent } from "@oh-my-pi/pi-agent-core";
import * as compactionModule from "@oh-my-pi/pi-agent-core/compaction";
import * as ai from "@oh-my-pi/pi-ai";
import type { AssistantMessage, Message, Model } from "@oh-my-pi/pi-ai";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { HIDDEN_TOOLS, type ToolSession } from "@oh-my-pi/pi-coding-agent/tools";
import { SnapcompactRecallTool } from "@oh-my-pi/pi-coding-agent/tools/snapcompact-recall";
import { TempDir } from "@oh-my-pi/pi-utils";
import * as snapcompact from "@oh-my-pi/snapcompact";

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
	let authStorage: AuthStorage | undefined;
	let tempDir: TempDir | undefined;

	afterEach(async () => {
		try {
			await session?.dispose();
		} finally {
			authStorage?.close();
			await tempDir?.remove();
			vi.restoreAllMocks();
			session = undefined;
			authStorage = undefined;
			tempDir = undefined;
		}
	});

	async function createHarness(
		options: { visionRole?: string | null; seedMessages?: Message[] } = {},
	): Promise<Harness> {
		const activeModel = getBundledModel("aimlapi", "alibaba/qwen3-coder-480b-a35b-instruct");
		if (!activeModel) throw new Error("Expected bundled text-only model");
		expect(activeModel.input).not.toContain("image");

		tempDir = TempDir.createSync("@pi-snapcompact-recall-");
		authStorage = await AuthStorage.create(path.join(tempDir.path(), "auth.db"));
		authStorage.setRuntimeApiKey("aimlapi", "test-key");
		const modelRegistry = new ModelRegistry(authStorage);

		const agent = new Agent({
			initialState: { model: activeModel, systemPrompt: ["Test"], tools: [], messages: [] },
		});
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
		const sessionRef: AgentSession = new AgentSession({
			agent,
			sessionManager,
			settings,
			modelRegistry,
			createSnapcompactRecallTool: async () => (await HIDDEN_TOOLS.snapcompact_recall(toolSession)) ?? null,
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
});
