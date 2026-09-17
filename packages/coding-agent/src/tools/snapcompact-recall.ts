import {
	instrumentedCompleteSimple,
	resolveTelemetry,
	type AgentTool,
	type AgentToolResult,
} from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, ImageContent, TextContent } from "@oh-my-pi/pi-ai";
import { type } from "@oh-my-pi/omptype";
import * as snapcompact from "@oh-my-pi/snapcompact";
import { prompt } from "@oh-my-pi/pi-utils";
import { extractTextContent } from "../commit/utils";
import { getLatestCompactionEntry } from "../session/session-context";
import { resolveSnapcompactVisionModel } from "../session/role-models";
import type { ToolSession } from ".";
import { ToolError } from "./tool-errors";
import description from "../prompts/tools/snapcompact-recall.md" with { type: "text" };
import systemPrompt from "../prompts/tools/snapcompact-recall-system.md" with { type: "text" };
import requestPrompt from "../prompts/tools/snapcompact-recall-request.md" with { type: "text" };

const schema = type({
	query: type("string").describe("A self-contained question about one exact archived detail"),
	"+": "reject",
});
type Params = typeof schema.infer;

export class SnapcompactRecallTool implements AgentTool<typeof schema> {
	readonly name = "snapcompact_recall";
	readonly approval = "read" as const;
	readonly label = "Snapcompact recall";
	readonly summary = "Ask the vision sidecar about an exact archived detail";
	readonly description = description;
	readonly parameters = schema;
	readonly strict = true;
	readonly intent = "omit" as const;
	readonly loadMode = "discoverable" as const;

	constructor(private readonly session: ToolSession) {}

	async execute(_toolCallId: string, params: Params, signal?: AbortSignal): Promise<AgentToolResult> {
		const session = this.session;
		const modelRegistry = session.modelRegistry;
		if (!modelRegistry) throw new ToolError("Model registry is unavailable for snapcompact recall.");
		const query = params.query.trim();
		if (!query) throw new ToolError("snapcompact_recall requires a non-empty self-contained query.");
		const manager = session.sessionManager;
		if (!manager) throw new ToolError("Snapcompact recall is unavailable without a session manager.");
		const expectedId = session.getSessionId?.();
		if (expectedId !== undefined && manager.getSessionId && manager.getSessionId() !== expectedId) {
			throw new ToolError("Snapcompact recall session is no longer current.");
		}
		const compaction = getLatestCompactionEntry(manager.getBranch());
		if (!compaction) throw new ToolError("No snapcompact archive is available for recall.");
		const archive = snapcompact.getPreservedArchive(compaction.preserveData);
		if (!archive || archive.frames.length < 1)
			throw new ToolError("No image-bearing snapcompact archive is available for recall.");
		const model = resolveSnapcompactVisionModel(session.settings, modelRegistry, session.getActiveModel?.());
		if (!model) throw new ToolError("No explicitly configured vision model is available for snapcompact recall.");
		const context = manager.buildSessionContext({ includeSnapcompactFrames: true });
		const summary = context.messages.find(
			message => message.role === "compactionSummary" && message.summary === compaction.summary,
		) as { summary: string; blocks?: (TextContent | ImageContent)[] } | undefined;
		if (!summary?.blocks) throw new ToolError("The snapcompact archive could not be reconstructed for recall.");
		const imageBlocks = summary.blocks.filter((block): block is ImageContent => block.type === "image");
		if (imageBlocks.length !== archive.frames.length) {
			throw new ToolError(
				"Snapcompact recall found incomplete frame data; rerun snapcompact before recalling archived details.",
			);
		}
		const frameBytes = imageBlocks.reduce((total, block) => total + block.data.length, 0);
		if (
			imageBlocks.length > snapcompact.providerImageBudget(model.provider) ||
			frameBytes > snapcompact.FRAME_DATA_BYTES_BUDGET
		) {
			throw new ToolError(
				"The snapcompact archive is too large for the configured vision model; rerun snapcompact before recalling archived details.",
			);
		}
		let response: AssistantMessage;
		try {
			response = await instrumentedCompleteSimple(
				model,
				{
					systemPrompt: [prompt.render(systemPrompt)],
					messages: [
						{
							role: "user",
							content: [
								{ type: "text", text: compaction.summary },
								...summary.blocks,
								{ type: "text", text: prompt.render(requestPrompt, { query }) },
							],
							timestamp: Date.now(),
						},
					],
				},
				{
					apiKey: modelRegistry.resolver(model, expectedId ?? undefined),
					signal,
					reasoning: undefined,
				},
				{
					telemetry: resolveTelemetry(session.getTelemetry?.(), expectedId ?? undefined),
					oneshotKind: "snapcompact_recall",
				},
			);
		} catch (error) {
			if (error instanceof Error && (error.name === "AbortError" || error.name === "TimeoutError"))
				throw new ToolError(error.message);
			throw error;
		}
		if (response.stopReason === "error")
			throw new ToolError(response.errorMessage ?? "Snapcompact recall request failed.");
		if (response.stopReason === "aborted") throw new ToolError("Snapcompact recall request aborted.");
		const text = extractTextContent(response);
		if (!text) throw new ToolError("Vision model returned no text output for snapcompact recall.");
		return { content: [{ type: "text", text }] };
	}
}
