import { afterAll, afterEach, beforeAll, describe, expect, it } from "bun:test";
import * as path from "node:path";
import { Effort, type Model } from "@oh-my-pi/pi-ai";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { cfgModelPresets } from "@oh-my-pi/pi-coding-agent/config/model-settings";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { createAgentSession } from "@oh-my-pi/pi-coding-agent/sdk";
import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";
import { parseArgs } from "../src/cli/args";

const SONNET = "anthropic/claude-sonnet-4-5";
const SONNET_46 = "anthropic/claude-sonnet-4-6";
const OPUS = "anthropic/claude-opus-4-5";

function bundled(selector: string) {
	const [, id] = selector.split("/");
	const model = getBundledModel("anthropic", id!);
	if (!model) throw new Error(`missing bundled model ${selector}`);
	return model;
}

describe("--preset CLI flag", () => {
	let fixtureDir: TempDir;
	let authStorage: AuthStorage;
	let modelRegistry: ModelRegistry;
	const sessions: AgentSession[] = [];
	const tempDirs: TempDir[] = [];

	beforeAll(async () => {
		fixtureDir = TempDir.createSync("@pi-cli-preset-fixture-");
		authStorage = await AuthStorage.create(path.join(fixtureDir.path(), "auth.db"));
		authStorage.keys.setRuntime("anthropic", "test-key");
		modelRegistry = new ModelRegistry(authStorage, path.join(fixtureDir.path(), "models.yml"));
		modelRegistry.registerProvider("preset-thinking", {
			api: "openai-completions",
			apiKey: "test-key",
			baseUrl: "https://example.com/v1",
			models: [
				{
					id: "default-level-model",
					name: "Default Level Model",
					reasoning: true,
					thinking: { mode: "effort", efforts: [Effort.Low, Effort.High], defaultLevel: Effort.High },
					input: ["text"],
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: 128_000,
					maxTokens: 8_192,
				},
			],
		});
	});

	afterEach(async () => {
		for (const session of sessions.splice(0)) await session.dispose();
		for (const dir of tempDirs.splice(0)) dir.removeSync();
	});

	afterAll(() => {
		modelRegistry.unregisterProvider("preset-thinking");
		authStorage.close();
		fixtureDir.removeSync();
	});

	/** Isolated settings with a saved preset `p` whose default role is OPUS. */
	function presetSettings() {
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		cfgModelPresets.setEntry(settings, "p", {
			modelRoles: { default: OPUS, smol: SONNET_46 },
		});
		return settings;
	}

	async function startSession(
		options: {
			settings?: Settings;
			preset?: string;
			model?: Model;
			sessionManager?: SessionManager;
		} = {},
	) {
		const cwd = fixtureDir.path();
		const result = await createAgentSession({
			cwd,
			agentDir: fixtureDir.path(),
			sessionManager: options.sessionManager ?? SessionManager.inMemory(cwd),
			authStorage,
			modelRegistry,
			settings: options.settings ?? presetSettings(),
			disableExtensionDiscovery: true,
			skills: [],
			promptTemplates: [],
			slashCommands: [],
			enableMCP: false,
			enableLsp: false,
			skipPythonPreflight: true,
			...options,
		});
		sessions.push(result.session);
		return result.session;
	}

	it("parses --preset into Args.preset", () => {
		expect(parseArgs(["--preset", "p"]).preset).toBe("p");
	});

	it("seeds the session activePreset and its roles at startup", async () => {
		const session = await startSession({ preset: "p" });

		expect(session.activePreset).toBe("p");
		// The preset's default role selects the session model.
		expect(session.model?.id).toBe("claude-opus-4-5");
		// The preset fills the remaining roles.
		expect(session.effectiveModelRole("smol")).toBe(SONNET_46);
	});

	it("applies a preset defaultThinkingLevel over the selected model default", async () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", "preset-thinking/default-level-model");
		cfgModelPresets.setEntry(settings, "p", {
			modelRoles: { default: "preset-thinking/default-level-model" },
			defaultThinkingLevel: Effort.Low,
		});

		const session = await startSession({ settings, preset: "p" });

		expect(session.thinkingLevel).toBe(Effort.Low);
	});

	it("rejects an unknown preset at startup, naming the available presets", async () => {
		await expect(
			createAgentSession({
				cwd: fixtureDir.path(),
				agentDir: fixtureDir.path(),
				sessionManager: SessionManager.inMemory(fixtureDir.path()),
				authStorage,
				modelRegistry,
				settings: presetSettings(),
				disableExtensionDiscovery: true,
				skills: [],
				promptTemplates: [],
				slashCommands: [],
				enableMCP: false,
				enableLsp: false,
				skipPythonPreflight: true,
				preset: "nope",
			}),
		).rejects.toThrow(`Unknown model preset "nope". Available presets: p`);
	});

	it("lets an explicit --model win over the preset's default role", async () => {
		const session = await startSession({ preset: "p", model: bundled(SONNET) });

		expect(session.activePreset).toBe("p");
		expect(session.model?.id).toBe("claude-sonnet-4-5");
		expect(session.effectiveModelRole("smol")).toBe(SONNET_46);
	});

	it("wins over the preset a resumed session replayed and journals the switch", async () => {
		const settings = presetSettings();
		cfgModelPresets.setEntry(settings, "q", { modelRoles: { default: SONNET_46 } });
		const sessionManager = SessionManager.inMemory(fixtureDir.path());
		sessionManager.appendPresetChange("q");

		const session = await startSession({ settings, preset: "p", sessionManager });

		expect(session.activePreset).toBe("p");
		expect(sessionManager.buildSessionContext().activePreset).toBe("p");
	});
});
