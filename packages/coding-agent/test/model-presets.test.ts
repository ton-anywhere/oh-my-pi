import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { Agent } from "@oh-my-pi/pi-agent-core";
import { Effort } from "@oh-my-pi/pi-ai";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import {
	acquireModelRoleMutation,
	applyModelPreset,
	deleteModelPreset,
	getModelPreset,
	getModelPresetNames,
	formatModelPresetSwitch,
	isCleanModelPresetSwitch,
	modelPresetSavedMessage,
	modelPresetShadowOwner,
	saveModelPreset,
} from "@oh-my-pi/pi-coding-agent/config/model-presets";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { cfgModelPresets } from "@oh-my-pi/pi-coding-agent/config/model-settings";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { cfgDefaultThinkingLevel } from "@oh-my-pi/pi-coding-agent/session/settings";
import { TempDir } from "@oh-my-pi/pi-utils";

const SONNET = "anthropic/claude-sonnet-4-5";
const SONNET_46 = "anthropic/claude-sonnet-4-6";
const OPUS = "anthropic/claude-opus-4-5";

function bundled(selector: string) {
	const [, id] = selector.split("/");
	const model = getBundledModel("anthropic", id!);
	if (!model) throw new Error(`missing bundled model ${selector}`);
	return model;
}

describe("model presets", () => {
	let fixtureDir: TempDir;
	let authStorage: AuthStorage;
	let modelRegistry: ModelRegistry;
	const sessions: AgentSession[] = [];
	const tempDirs: TempDir[] = [];

	beforeAll(async () => {
		fixtureDir = TempDir.createSync("@pi-model-presets-fixture-");
		authStorage = await AuthStorage.create(path.join(fixtureDir.path(), "auth.db"));
		authStorage.keys.setRuntime("anthropic", "test-key");
		modelRegistry = new ModelRegistry(authStorage, path.join(fixtureDir.path(), "models.yml"));
	});

	afterEach(async () => {
		vi.restoreAllMocks();
		for (const session of sessions.splice(0)) await session.dispose();
		for (const dir of tempDirs.splice(0)) dir.removeSync();
	});

	afterAll(() => {
		authStorage.close();
		fixtureDir.removeSync();
	});

	function createSession(settings: Settings, initialModel = SONNET, thinkingLevel: Effort = Effort.High) {
		const agent = new Agent({
			initialState: { model: bundled(initialModel), systemPrompt: ["Test"], tools: [], messages: [], thinkingLevel },
		});
		const session = new AgentSession({ agent, sessionManager: SessionManager.inMemory(), settings, modelRegistry });
		sessions.push(session);
		return session;
	}

	/** Settings with a real project layer and an optional `--config` overlay. */
	async function projectSettings(options: { project?: string; overlay?: string; storage?: "global" | "project" }) {
		const dir = TempDir.createSync("@pi-model-presets-project-");
		tempDirs.push(dir);
		const projectDir = path.join(dir.path(), "project");
		fs.mkdirSync(path.join(projectDir, ".omp"), { recursive: true });
		if (options.project) fs.writeFileSync(path.join(projectDir, ".omp", "config.yml"), options.project);
		const configFiles: string[] = [];
		if (options.overlay) {
			const overlayPath = path.join(dir.path(), "overlay.yml");
			fs.writeFileSync(overlayPath, options.overlay);
			configFiles.push(overlayPath);
		}
		return Settings.loadIsolated({
			cwd: projectDir,
			agentDir: dir.path(),
			inMemory: true,
			configFiles,
			overrides: options.storage ? { modelRoleStorage: options.storage } : {},
		});
	}

	it("effectiveModelRole resolves explicit override, active preset, then base", async () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		settings.setModelRole("smol", SONNET_46);
		cfgModelPresets.setEntry(settings, "snap", { modelRoles: { default: OPUS, slow: "p/slow" } });
		const session = createSession(settings);

		// No preset active: the base decides everything.
		expect(session.effectiveModelRole("default")).toBe(SONNET);
		expect(session.effectiveModelRole("smol")).toBe(SONNET_46);
		expect(session.effectiveModelRole("plan")).toBeUndefined();

		const result = await applyModelPreset(settings, session, "snap");
		expect(result.kind).toBe("switched");
		// The preset fills its roles; the base fills what it leaves out.
		expect(session.effectiveModelRole("default")).toBe(OPUS);
		expect(session.effectiveModelRole("slow")).toBe("p/slow");
		expect(session.effectiveModelRole("smol")).toBe(SONNET_46);

		// An explicit session selector outranks the preset's own role.
		session.setSessionRoleOverride("default", SONNET);
		expect(session.effectiveModelRole("default")).toBe(SONNET);
		expect(session.effectiveModelRole("slow")).toBe("p/slow");
	});

	it("effectiveDefaultThinkingLevel takes the preset's level when it records one", async () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		cfgDefaultThinkingLevel.set(settings, Effort.High);
		cfgModelPresets.setEntry(settings, "snap", {
			modelRoles: { default: OPUS },
			defaultThinkingLevel: Effort.Low,
		});
		const session = createSession(settings, SONNET, Effort.High);

		expect(session.effectiveDefaultThinkingLevel()).toBe(Effort.High);

		const result = await applyModelPreset(settings, session, "snap");
		expect(result.kind).toBe("switched");
		expect(session.effectiveDefaultThinkingLevel()).toBe(Effort.Low);
	});

	it("switching replaces the whole role record and applies the preset's thinking level live", async () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", OPUS);
		settings.setModelRole("plan", SONNET_46);
		cfgDefaultThinkingLevel.set(settings, Effort.Low);
		saveModelPreset(settings, "focus");

		settings.setModelRole("default", SONNET);
		settings.setModelRole("plan", undefined);
		settings.setModelRole("smol", SONNET_46);
		cfgDefaultThinkingLevel.set(settings, Effort.High);
		const session = createSession(settings, SONNET, Effort.High);

		const result = await applyModelPreset(settings, session, "focus");

		expect(result.kind).toBe("switched");
		// The shared config is never written; only the session is affected.
		expect({ ...settings.getModelRoles() }).toEqual({ default: SONNET, smol: SONNET_46 });
		expect(cfgDefaultThinkingLevel.get(settings)).toBe(Effort.High);
		expect(session.activePreset).toBe("focus");
		expect(session.model?.id).toBe("claude-opus-4-5");
		// The live agent state (what #runLoop sends the next request with) carries the switch.
		expect(session.agent.state.model?.id).toBe("claude-opus-4-5");
		// setModel alone keeps the previous `high`; the preset's `low` must win.
		expect(session.thinkingLevel).toBe(Effort.Low);
	});

	it("switching a preset records it per-session without writing shared config", async () => {
		const dir = TempDir.createSync("@pi-model-presets-isolation-");
		tempDirs.push(dir);
		const agentDir = path.join(dir.path(), "agent");
		fs.mkdirSync(agentDir, { recursive: true });
		fs.writeFileSync(
			path.join(agentDir, "config.yml"),
			`modelRoles:\n  default: ${SONNET}\nmodelPresets:\n  p:\n    modelRoles:\n      default: ${OPUS}\n`,
		);
		const load = () => Settings.loadIsolated({ cwd: dir.path(), agentDir });
		const settings = await load();
		const session = createSession(settings);

		const result = await applyModelPreset(settings, session, "p");

		expect(result.kind).toBe("switched");
		// The session records its active preset; the shared config is untouched.
		expect(session.activePreset).toBe("p");
		expect(settings.getModelRole("default")).toBe(SONNET);
		// A fresh Settings for the same scope still resolves its own base roles.
		const independent = await load();
		expect(independent.getModelRole("default")).toBe(SONNET);
	});

	it("an explicit selector level on the preset's default beats its defaultThinkingLevel", async () => {
		const settings = Settings.isolated();
		cfgModelPresets.setEntry(settings, "deep", {
			modelRoles: { default: `${OPUS}:medium` },
			defaultThinkingLevel: Effort.Low,
		});
		const session = createSession(settings, SONNET, Effort.High);

		const result = await applyModelPreset(settings, session, "deep");

		expect(result.kind).toBe("switched");
		expect(session.thinkingLevel).toBe(Effort.Medium);
	});

	it("in project storage, the preset shadows the project layer and the shared config is untouched", async () => {
		const settings = await projectSettings({
			project: `modelRoles:\n  default: ${SONNET}\n  slow: ${SONNET_46}\n`,
			storage: "project",
		});
		cfgModelPresets.setEntry(settings, "lean", { modelRoles: { default: OPUS } });
		const session = createSession(settings);

		const result = await applyModelPreset(settings, session, "lean");

		expect(result.kind).toBe("switched");
		if (result.kind !== "switched") throw new Error(`expected "switched", got ${result.kind}`);
		// The preset outranks the project layer for the roles it defines; roles it
		// leaves out keep their base assignment. The shared config is untouched.
		expect(result.shadowed).toEqual([{ role: "slow", expected: undefined, actual: SONNET_46, source: "project" }]);
		expect(session.effectiveModelRole("default")).toBe(OPUS);
		expect({ ...settings.getModelRoles() }).toEqual({ default: SONNET, slow: SONNET_46 });
		expect(settings.getProjectModelRole("slow")).toBe(SONNET_46);
		expect(session.model?.id).toBe("claude-opus-4-5");
	});

	it("a preset shadows --model style runtime role overrides, which stay recorded", async () => {
		const settings = Settings.isolated();
		settings.overrideModelRoles({ default: SONNET, smol: SONNET_46 });
		cfgModelPresets.setEntry(settings, "opus", { modelRoles: { default: OPUS } });
		const session = createSession(settings);

		const result = await applyModelPreset(settings, session, "opus");

		expect(result.kind).toBe("switched");
		if (result.kind !== "switched") throw new Error(`expected "switched", got ${result.kind}`);
		expect(result.shadowed).toEqual([{ role: "smol", expected: undefined, actual: SONNET_46, source: "runtime" }]);
		expect(session.effectiveModelRole("default")).toBe(OPUS);
		expect({ ...settings.getModelRoles() }).toEqual({ default: SONNET, smol: SONNET_46 });
		expect(session.model?.id).toBe("claude-opus-4-5");
	});

	it("the preset shadows overlay and project layers alike; the global layer stays empty", async () => {
		const settings = await projectSettings({
			project: `modelRoles:\n  slow: ${SONNET_46}\n`,
			overlay: `modelRoles:\n  default: ${SONNET}\n`,
		});
		cfgModelPresets.setEntry(settings, "opus", { modelRoles: { default: OPUS } });
		const session = createSession(settings, SONNET);

		const result = await applyModelPreset(settings, session, "opus");

		if (result.kind !== "switched") throw new Error(`expected switched, got ${result.kind}`);
		expect(result.shadowed).toEqual([{ role: "slow", expected: undefined, actual: SONNET_46, source: "project" }]);
		// The live model follows the preset's default, which now outranks the overlay.
		expect(session.model?.id).toBe("claude-opus-4-5");
		// The shared config is never written; the global layer stays empty.
		expect(settings.getGlobalModelRole("default")).toBeUndefined();
	});

	it("never resolves inherited object keys as presets", async () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		const session = createSession(settings);

		for (const name of ["toString", "constructor", "__proto__", "hasOwnProperty"]) {
			expect((await applyModelPreset(settings, session, name)).kind).toBe("missing");
			expect(deleteModelPreset(settings, name)).toBe("missing");
		}
		expect(settings.getModelRole("default")).toBe(SONNET);
	});

	it("rejects a malformed hand-written preset without touching roles", async () => {
		const settings = await projectSettings({
			overlay: `modelPresets:\n  broken:\n    modelRoles: ${OPUS}\n`,
		});
		settings.setModelRole("default", SONNET);
		const session = createSession(settings);

		const result = await applyModelPreset(settings, session, "broken");

		expect(result.kind).toBe("invalid");
		expect(settings.getModelRole("default")).toBe(SONNET);
	});

	it("leaves settings untouched when the preset's default model is unavailable", async () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		cfgModelPresets.setEntry(settings, "gone", { modelRoles: { default: "nosuch/provider-model" } });
		const session = createSession(settings);

		const result = await applyModelPreset(settings, session, "gone");

		expect(result.kind).toBe("unavailable");
		expect(settings.getModelRole("default")).toBe(SONNET);
		expect(session.model?.id).toBe("claude-sonnet-4-5");
	});

	it("reports a failed live switch instead of success", async () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		cfgModelPresets.setEntry(settings, "opus", { modelRoles: { default: OPUS } });
		const session = createSession(settings);
		vi.spyOn(session.agent, "setModel").mockImplementation(() => {
			throw new Error("metadata refresh failed");
		});

		const result = await applyModelPreset(settings, session, "opus");

		expect(result).toMatchObject({ kind: "failed", reason: "metadata refresh failed" });
		// A failed switch records nothing: the prior preset and model are retained.
		expect(session.activePreset).toBeUndefined();
		expect(session.model?.id).toBe("claude-sonnet-4-5");
	});

	it("saves only the named entry globally and never copies project presets", async () => {
		const settings = await projectSettings({
			project: `modelPresets:\n  team:\n    modelRoles:\n      default: ${SONNET}\n`,
		});
		settings.setModelRole("default", OPUS);

		saveModelPreset(settings, "mine");

		expect(Object.keys(settings.getGlobalSettings().modelPresets ?? {})).toEqual(["mine"]);
		expect(getModelPresetNames(settings)).toEqual(["mine", "team"]);
		expect(deleteModelPreset(settings, "team")).toBe("project");
		expect(deleteModelPreset(settings, "mine")).toBe("deleted");
		expect(getModelPresetNames(settings)).toEqual(["team"]);
	});

	it("resolves an alias default against the preset's own roles, not the current ones", async () => {
		// Current `slow` is broken but the preset redefines it: the switch must pass pre-validation.
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		settings.setModelRole("slow", "nosuch/gone-model");
		cfgModelPresets.setEntry(settings, "alias", {
			modelRoles: { default: "@slow", slow: SONNET_46 },
		});
		const session = createSession(settings);

		const result = await applyModelPreset(settings, session, "alias");

		expect(result.kind).toBe("switched");
		// The shared config is never written; the broken `slow` role stays as-is.
		expect(settings.getModelRole("slow")).toBe("nosuch/gone-model");
		// The live switch resolved the alias against the preset's own roles.
		expect(session.model?.id).toBe("claude-sonnet-4-6");
	});

	it("refuses an alias default the preset itself breaks, leaving settings untouched", async () => {
		// Current `slow` is fine but the preset redefines it as broken: unavailable, nothing written.
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		settings.setModelRole("slow", SONNET_46);
		cfgModelPresets.setEntry(settings, "alias", {
			modelRoles: { default: "@slow", slow: "nosuch/gone-model" },
		});
		const session = createSession(settings);

		const result = await applyModelPreset(settings, session, "alias");

		expect(result.kind).toBe("unavailable");
		expect(settings.getModelRole("slow")).toBe(SONNET_46);
		expect(settings.getModelRole("default")).toBe(SONNET);
	});

	it("reports a same-name project preset instead of a false deletion", async () => {
		const settings = await projectSettings({
			project: `modelPresets:\n  team:\n    modelRoles:\n      default: ${SONNET}\n`,
		});
		cfgModelPresets.setEntry(settings, "team", { modelRoles: { default: OPUS } });

		expect(deleteModelPreset(settings, "team")).toBe("project");
		expect(getModelPresetNames(settings)).toEqual(["team"]);
	});

	it("leaves the session level alone for an :inherit default suffix", async () => {
		const settings = Settings.isolated();
		cfgModelPresets.setEntry(settings, "inh", { modelRoles: { default: `${OPUS}:inherit` } });
		const session = createSession(settings);
		const thinkingSpy = vi.spyOn(session, "setThinkingLevel");

		const result = await applyModelPreset(settings, session, "inh");

		expect(result.kind).toBe("switched");
		expect(thinkingSpy).not.toHaveBeenCalled();
		expect(session.model?.id).toBe("claude-opus-4-5");
	});

	it("resolves a same-name preset whole from the project layer, never merged", async () => {
		const settings = await projectSettings({
			project: `modelPresets:\n  deep:\n    modelRoles:\n      default: ${OPUS}\n`,
		});
		settings.setModelRole("slow", SONNET_46);
		cfgModelPresets.setEntry(settings, "deep", {
			modelRoles: { default: SONNET, smol: SONNET_46 },
			defaultThinkingLevel: Effort.High,
		});
		const session = createSession(settings);

		const result = await applyModelPreset(settings, session, "deep");

		expect(result.kind).toBe("switched");
		// The project `deep` sets only `default`: the global `deep`'s `smol` must not leak in.
		// The shared config is never written; the stale `slow` role stays as-is.
		expect({ ...settings.getModelRoles() }).toEqual({ slow: SONNET_46 });
		expect(session.model?.id).toBe("claude-opus-4-5");
	});

	it("refuses a preset with no default and no authed model before writing", async () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		cfgModelPresets.setEntry(settings, "auto", { modelRoles: {} });
		const session = createSession(settings);
		vi.spyOn(session, "getAvailableModels").mockReturnValue([bundled(OPUS)]);
		vi.spyOn(modelRegistry, "hasConfiguredAuth").mockReturnValue(false);

		const result = await applyModelPreset(settings, session, "auto");

		expect(result.kind).toBe("unavailable");
		expect(settings.getModelRole("default")).toBe(SONNET);
		expect(session.model?.id).toBe("claude-sonnet-4-5");
	});

	it("reports when a higher layer still defines a just-saved preset name", async () => {
		const settings = await projectSettings({
			project: `modelPresets:\n  team:\n    modelRoles:\n      default: ${SONNET}\n`,
		});
		settings.setModelRole("default", OPUS);

		saveModelPreset(settings, "team");

		expect(modelPresetShadowOwner(settings, "team")).toBe("project");
		// The effective preset is still the project's, not the just-saved global entry.
		expect(settings.getModelRole("default")).toBe(OPUS);
		const lookup = getModelPreset(settings, "team");
		expect(lookup).toMatchObject({ kind: "found", preset: { modelRoles: { default: SONNET } } });
		expect(modelPresetSavedMessage(settings, "team")).toBe(
			'Saved model preset "team" to the global config, but the project config still defines a preset of the same name, which takes precedence',
		);
	});

	it("serializes concurrent default-role mutations in acquisition order", async () => {
		const order: string[] = [];
		const first = await acquireModelRoleMutation();
		const secondPromise = acquireModelRoleMutation();
		let secondResolved = false;
		void secondPromise.then(release => {
			secondResolved = true;
			order.push("second");
			release();
		});
		// The mutex is promise-chaining only (no timers), so draining microtasks
		// deterministically proves the second acquisition is still parked.
		await Promise.resolve();
		await Promise.resolve();
		expect(secondResolved).toBe(false);
		order.push("first");
		first();
		await secondPromise;
		expect(order).toEqual(["first", "second"]);
	});

	it("applies the preset's thinking level live, even when a base layer sets a different one", async () => {
		const settings = await projectSettings({ project: "defaultThinkingLevel: high\n" });
		cfgModelPresets.setEntry(settings, "cheap", {
			modelRoles: { default: OPUS },
			defaultThinkingLevel: Effort.Low,
		});
		const session = createSession(settings, SONNET, Effort.High);

		const result = await applyModelPreset(settings, session, "cheap");

		if (result.kind !== "switched") throw new Error(`expected switched, got ${result.kind}`);
		expect(session.thinkingLevel).toBe(Effort.Low);
		expect(result.shadowed).toEqual([]);
		expect(formatModelPresetSwitch("cheap", result)).toBe(
			'Switched to preset "cheap" (anthropic/claude-opus-4-5 · low)',
		);
	});

	it("does not list a preset a --config overlay tombstones, but keeps one a project null falls through", async () => {
		const tombstoned = await projectSettings({ overlay: "modelPresets:\n  deep: null\n" });
		cfgModelPresets.setEntry(tombstoned, "deep", { modelRoles: { default: OPUS } });
		cfgModelPresets.setEntry(tombstoned, "fast", { modelRoles: { default: SONNET } });
		expect(getModelPresetNames(tombstoned)).toEqual(["fast"]);

		const fallsThrough = await projectSettings({ project: "modelPresets:\n  deep: null\n" });
		cfgModelPresets.setEntry(fallsThrough, "deep", { modelRoles: { default: OPUS } });
		expect(getModelPresetNames(fallsThrough)).toEqual(["deep"]);
	});

	it("holds the model-role mutation lock for the whole apply", async () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		cfgModelPresets.setEntry(settings, "opus", { modelRoles: { default: OPUS } });
		const session = createSession(settings);

		const release = await acquireModelRoleMutation();
		const applying = applyModelPreset(settings, session, "opus");
		// Promise-chaining mutex, no timers: drained microtasks prove the apply is parked.
		for (let i = 0; i < 5; i++) await Promise.resolve();
		expect(settings.getModelRole("default")).toBe(SONNET);
		expect(session.model?.id).toBe("claude-sonnet-4-5");

		release();
		expect((await applying).kind).toBe("switched");
		// The shared config is never written; only the session is affected.
		expect(settings.getModelRole("default")).toBe(SONNET);
		expect(session.model?.id).toBe("claude-opus-4-5");
	});

	it("an explicit session override beats the preset's own role and is reported as shadowed", async () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		cfgModelPresets.setEntry(settings, "opus", { modelRoles: { default: OPUS, smol: SONNET_46 } });
		const session = createSession(settings);
		session.setSessionRoleOverride("default", SONNET);

		const result = await applyModelPreset(settings, session, "opus");

		expect(result.kind).toBe("switched");
		if (result.kind !== "switched") throw new Error(`expected switched, got ${result.kind}`);
		expect(result.shadowed).toEqual([{ role: "default", expected: OPUS, actual: SONNET, source: "session" }]);
		expect(isCleanModelPresetSwitch(result)).toBe(false);
		// The explicit pick wins the live model; the preset's other roles apply.
		expect(session.model?.id).toBe("claude-sonnet-4-5");
		expect(session.effectiveModelRole("smol")).toBe(SONNET_46);
	});

	it("consecutive switches do not leak the previous preset's roles", async () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		cfgModelPresets.setEntry(settings, "a", { modelRoles: { default: OPUS, smol: SONNET_46 } });
		cfgModelPresets.setEntry(settings, "b", { modelRoles: { default: SONNET_46 } });
		const session = createSession(settings);

		expect((await applyModelPreset(settings, session, "a")).kind).toBe("switched");
		expect(session.effectiveModelRole("smol")).toBe(SONNET_46);

		const result = await applyModelPreset(settings, session, "b");

		expect(result.kind).toBe("switched");
		if (result.kind !== "switched") throw new Error(`expected switched, got ${result.kind}`);
		// `a`'s `smol` must not survive the switch to `b`.
		expect(session.effectiveModelRole("smol")).toBeUndefined();
		expect(session.effectiveModelRole("default")).toBe(SONNET_46);
		expect(session.model?.id).toBe("claude-sonnet-4-6");
	});

	it("a clean switch reports no shadowed roles", async () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", SONNET);
		cfgModelPresets.setEntry(settings, "opus", { modelRoles: { default: OPUS } });
		const session = createSession(settings);

		const result = await applyModelPreset(settings, session, "opus");

		expect(result.kind).toBe("switched");
		if (result.kind !== "switched") throw new Error(`expected switched, got ${result.kind}`);
		expect(result.shadowed).toEqual([]);
		expect(isCleanModelPresetSwitch(result)).toBe(true);
		expect(formatModelPresetSwitch("opus", result)).toBe('Switched to preset "opus" (anthropic/claude-opus-4-5)');
	});

	it("the next prompt streams on the switched model", async () => {
		const settings = Settings.isolated({ "compaction.enabled": false });
		settings.setModelRole("default", SONNET);
		cfgModelPresets.setEntry(settings, "focus", { modelRoles: { default: OPUS } });

		const streamed: string[] = [];
		const inner = createMockModel({ responses: [{ content: ["Done"] }] });
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model: bundled(SONNET), systemPrompt: ["Test"], tools: [], messages: [] },
			streamFn: (model, context, options) => {
				streamed.push(model.id);
				return inner.stream(model, context, options);
			},
		});
		const session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings,
			modelRegistry,
		});
		sessions.push(session);

		expect((await applyModelPreset(settings, session, "focus")).kind).toBe("switched");
		await session.prompt("hello");

		expect(streamed).toEqual(["claude-opus-4-5"]);
	});

	it("rebinds a live preset alias edit in the next stream", async () => {
		const settings = Settings.isolated({ "compaction.enabled": false });
		settings.setModelRole("default", SONNET);
		cfgModelPresets.setEntry(settings, "alias", {
			modelRoles: { default: "@slow", slow: SONNET_46 },
			defaultThinkingLevel: Effort.Low,
		});

		const streamed: Array<{ id: string; reasoning: Effort | undefined }> = [];
		const inner = createMockModel({ responses: [{ content: ["Done"] }] });
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model: bundled(SONNET), systemPrompt: ["Test"], tools: [], messages: [] },
			streamFn: (model, context, options) => {
				streamed.push({ id: model.id, reasoning: options?.reasoning });
				return inner.stream(model, context, options);
			},
		});
		const session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings,
			modelRegistry,
		});
		sessions.push(session);

		const rebound = Promise.withResolvers<void>();
		session.subscribe(event => {
			if (event.type === "thinking_level_changed" && event.thinkingLevel === Effort.High) rebound.resolve();
		});
		expect((await applyModelPreset(settings, session, "alias")).kind).toBe("switched");
		cfgModelPresets.setEntry(settings, "alias", {
			modelRoles: { default: "@slow", slow: OPUS },
			defaultThinkingLevel: Effort.High,
		});
		await rebound.promise;

		expect(session.agent.state.model?.id).toBe("claude-opus-4-5");
		expect(session.thinkingLevel).toBe(Effort.High);
		await session.prompt("hello");

		expect(streamed).toEqual([{ id: "claude-opus-4-5", reasoning: Effort.High }]);
	});

	it("restores a target preset alias during a session switch", async () => {
		const settings = Settings.isolated({ "compaction.enabled": false });
		settings.setModelRole("default", SONNET);
		cfgModelPresets.setEntry(settings, "alias", {
			modelRoles: { default: "@slow", slow: OPUS },
			defaultThinkingLevel: Effort.High,
		});
		const dir = TempDir.createSync("@pi-model-presets-switch-");
		tempDirs.push(dir);
		const sessionManager = SessionManager.create(dir.path(), dir.path());
		const session = new AgentSession({
			agent: new Agent({
				initialState: { model: bundled(SONNET), systemPrompt: ["Test"], tools: [], messages: [] },
			}),
			sessionManager,
			settings,
			modelRegistry,
		});
		sessions.push(session);

		const target = SessionManager.create(dir.path(), dir.path());
		target.appendModelChange(SONNET);
		target.appendThinkingLevelChange(Effort.Low, Effort.Low);
		target.appendPresetChange("alias");
		target.appendMessage({ role: "user", content: "target", timestamp: Date.now() });
		await target.ensureOnDisk();
		const targetFile = target.getSessionFile();
		if (!targetFile) throw new Error("Expected target session file");

		await expect(session.switchSession(targetFile)).resolves.toBe(true);

		expect(session.agent.state.model?.id).toBe("claude-opus-4-5");
		expect(session.thinkingLevel).toBe(Effort.High);
	});

	it("rebinds model and thinking when tree navigation enters another preset branch", async () => {
		const settings = Settings.isolated({ "compaction.enabled": false });
		settings.setModelRole("default", SONNET);
		cfgModelPresets.setEntry(settings, "low", {
			modelRoles: { default: SONNET_46 },
			defaultThinkingLevel: Effort.Low,
		});
		cfgModelPresets.setEntry(settings, "high", {
			modelRoles: { default: OPUS },
			defaultThinkingLevel: Effort.High,
		});
		const sessionManager = SessionManager.inMemory();
		const session = new AgentSession({
			agent: new Agent({
				initialState: { model: bundled(SONNET), systemPrompt: ["Test"], tools: [], messages: [] },
			}),
			sessionManager,
			settings,
			modelRegistry,
		});
		sessions.push(session);

		const root = sessionManager.appendMessage({ role: "user", content: "root", timestamp: Date.now() });
		sessionManager.appendPresetChange("low");
		const lowMessage = sessionManager.appendMessage({ role: "user", content: "low", timestamp: Date.now() });
		sessionManager.branch(root);
		sessionManager.appendPresetChange("high");
		const highMessage = sessionManager.appendMessage({ role: "user", content: "high", timestamp: Date.now() });

		await session.navigateTree(lowMessage);
		expect(session.agent.state.model?.id).toBe("claude-sonnet-4-6");
		expect(session.thinkingLevel).toBe(Effort.Low);

		await session.navigateTree(highMessage);
		expect(session.agent.state.model?.id).toBe("claude-opus-4-5");
		expect(session.thinkingLevel).toBe(Effort.High);
	});

	it("saveModelPreset through the session's settings reaches the shared config", async () => {
		const dir = TempDir.createSync("@pi-model-presets-persist-");
		tempDirs.push(dir);
		const agentDir = path.join(dir.path(), "agent");
		fs.mkdirSync(agentDir, { recursive: true });
		const load = () => Settings.loadIsolated({ cwd: dir.path(), agentDir });
		const settings = await load();
		settings.setModelRole("default", SONNET);
		await settings.flush();
		const session = createSession(settings);

		saveModelPreset(session.settings, "p2");
		await session.settings.flush();

		// An independent reload of the same scope sees the definition: the
		// session's settings are the durable base, not a write-intercepting overlay.
		const independent = await load();
		expect(getModelPresetNames(independent)).toEqual(["p2"]);
		expect(getModelPreset(independent, "p2")).toMatchObject({
			kind: "found",
			preset: { modelRoles: { default: SONNET } },
		});
	});
});
