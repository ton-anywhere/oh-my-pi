/**
 * Model presets: named snapshots of the role assignments (`modelRoles`) and
 * `defaultThinkingLevel`, saved and re-applied as a whole.
 *
 * Presets persist to the global config (`modelPresets.<name>`); a project config
 * may also define presets, which are listed and applied but never copied into
 * the global file. Switching a preset records it as per-session state and
 * re-resolves the session's roles from the definition — the shared config is
 * never written, so concurrent sessions on different presets do not clobber
 * each other. Roles still decided by a higher layer (a `--config` overlay or a
 * runtime `--model` override) win over the preset and are reported back.
 */
import type { Model } from "@oh-my-pi/pi-ai";
import { ThinkingLevel } from "@oh-my-pi/pi-agent-core";
import { THINKING_EFFORTS } from "@oh-my-pi/pi-catalog/effort";
import { AUTO_THINKING, type ConfiguredThinkingLevel, parseConfiguredThinkingLevel } from "@oh-my-pi/pi-tui/thinking";
import { isRecord } from "@oh-my-pi/pi-utils";
import type { AgentSession } from "../session/agent-session";
import { cfgDefaultThinkingLevel } from "../session/settings";
import { pickDefaultAvailableModel, resolveModelRoleValue } from "./model-resolver";
import { cfgModelPresets, cfgModelRoleStorage, type ModelPreset } from "./model-settings";
import type { Settings, SettingProvenance } from "./settings";

/** Preset names: a letter, then letters, digits, `-` or `_` (the model hub's role-name rule). */
const PRESET_NAME_PATTERN = /^[a-zA-Z][\w-]*$/;

export function isValidModelPresetName(name: string): boolean {
	return PRESET_NAME_PATTERN.test(name);
}

type PresetLookup = { kind: "found"; preset: ModelPreset } | { kind: "missing" } | { kind: "invalid"; reason: string };

/** Validate one raw `modelPresets` entry; hand-edited config can hold anything. */
function parseModelPreset(raw: unknown): ModelPreset | string {
	if (!isRecord(raw)) return "it is not a mapping";
	const roles = raw.modelRoles;
	if (!isRecord(roles)) return "`modelRoles` is missing or not a mapping";
	const modelRoles: Record<string, string> = {};
	for (const role of Object.keys(roles)) {
		const value = roles[role];
		if (typeof value !== "string" || value.trim() === "") return `role \`${role}\` is not a model selector`;
		modelRoles[role] = value;
	}
	const level = raw.defaultThinkingLevel;
	if (level === undefined) return { modelRoles };
	const thinking = typeof level === "string" ? parseConfiguredThinkingLevel(level) : undefined;
	if (thinking === undefined || !isDefaultThinkingLevel(thinking)) {
		return "`defaultThinkingLevel` is not a thinking level";
	}
	return { modelRoles, defaultThinkingLevel: thinking };
}

function isDefaultThinkingLevel(
	level: ConfiguredThinkingLevel,
): level is NonNullable<ModelPreset["defaultThinkingLevel"]> {
	return level === AUTO_THINKING || THINKING_EFFORTS.some(effort => effort === level);
}

/** Look a preset up whole from the highest layer owning its name — same-name entries never merge. */
export function getModelPreset(settings: Settings, name: string): PresetLookup {
	const owned = settings.getOwnedModelPreset(name);
	if (!owned) return { kind: "missing" };
	const parsed = parseModelPreset(owned.entry);
	return typeof parsed === "string" ? { kind: "invalid", reason: parsed } : { kind: "found", preset: parsed };
}

/**
 * Session `Settings` overlay for the active preset: role resolution uses the
 * 3-tier precedence per-role override (`roleOverrides`) → active preset role →
 * base `modelRoles`, and `defaultThinkingLevel` comes from the preset when it
 * records one. Passes the base through (no behavior change) when `activePreset`
 * is `undefined` or its lookup is not `found`.
 *
 * The child's `modelRoles` deep-merges per-key over the parent, so a role absent
 * from both the preset and the overrides falls through to its base value.
 */
export function buildPresetOverlay(
	base: Settings,
	activePreset: string | undefined,
	roleOverrides: Record<string, string>,
): Settings {
	if (!activePreset) return base.overlay({});
	const lookup = getModelPreset(base, activePreset);
	if (lookup.kind !== "found") return base.overlay({});
	const overrides: Record<string, unknown> = {
		modelRoles: { ...lookup.preset.modelRoles, ...roleOverrides },
	};
	if (lookup.preset.defaultThinkingLevel !== undefined) {
		overrides.defaultThinkingLevel = lookup.preset.defaultThinkingLevel;
	}
	return base.overlay(overrides);
}

/** Roles decided by a layer above the base: runtime `--model`/env overrides and the `--config` overlay. */
export function roleOverridesAboveBase(settings: Settings): Record<string, string> {
	const overrides: Record<string, string> = {};
	for (const [role, value] of Object.entries(settings.getModelRoles())) {
		const provenance = settings.getModelRoleProvenance(role);
		if (value !== undefined && (provenance === "runtime" || provenance === "overlay")) overrides[role] = value;
	}
	return overrides;
}

/** Saved preset names, sorted; names a `--config`/runtime `null` tombstone hides are left out. */
export function getModelPresetNames(settings: Settings): string[] {
	return Object.keys(cfgModelPresets.get(settings))
		.filter(name => settings.getOwnedModelPreset(name) !== undefined)
		.sort((a, b) => a.localeCompare(b));
}

/**
 * First saved preset (by name) the current setup matches: identical effective
 * role assignments and, when the preset records one, the same default thinking
 * level. Undefined when the roles were edited past every preset.
 */
export function findActiveModelPreset(settings: Settings): string | undefined {
	const roles = settings.getModelRoles();
	// Same filter as `saveModelPreset`: empty selectors are never stored.
	const roleIds = Object.keys(roles).filter(role => roles[role]);
	const thinking = cfgDefaultThinkingLevel.get(settings);
	return getModelPresetNames(settings).find(name => {
		const lookup = getModelPreset(settings, name);
		if (lookup.kind !== "found") return false;
		const { modelRoles, defaultThinkingLevel } = lookup.preset;
		if (defaultThinkingLevel !== undefined && defaultThinkingLevel !== thinking) return false;
		const presetIds = Object.keys(modelRoles);
		return (
			presetIds.length === roleIds.length &&
			presetIds.every(role => Object.hasOwn(roles, role) && roles[role] === modelRoles[role])
		);
	});
}

/**
 * Save the effective role assignments and `defaultThinkingLevel` as `name` in the
 * global config, overwriting a preset of the same name. Only this entry is
 * written, so presets defined by a project config are never copied globally.
 *
 * @throws Error when `name` is not a valid preset name.
 */
export function saveModelPreset(settings: Settings, name: string): ModelPreset {
	if (!isValidModelPresetName(name)) {
		throw new Error(`Invalid preset name "${name}": use a letter, then letters, digits, - or _`);
	}
	const modelRoles: Record<string, string> = {};
	for (const [role, selector] of Object.entries(settings.getModelRoles())) {
		if (selector) modelRoles[role] = selector;
	}
	const preset: ModelPreset = { modelRoles, defaultThinkingLevel: cfgDefaultThinkingLevel.get(settings) };
	cfgModelPresets.setEntry(settings, name, preset);
	return preset;
}

export type ModelPresetDeleteResult = "deleted" | "missing" | "project";

/**
 * Delete `name` from the global config. A preset only a project (or `--config`)
 * file defines is reported as `project`: it has to be removed from that file.
 * The same applies when another layer defines the same name: the global entry is
 * removed but the preset keeps listing, so `project` is reported instead of a
 * false `deleted`.
 */
export function deleteModelPreset(settings: Settings, name: string): ModelPresetDeleteResult {
	const global = settings.getGlobalSettings().modelPresets;
	if (isRecord(global) && Object.hasOwn(global, name)) {
		cfgModelPresets.setEntry(settings, name, undefined);
		return settings.getOwnedModelPreset(name) ? "project" : "deleted";
	}
	return settings.getOwnedModelPreset(name) ? "project" : "missing";
}

/**
 * Layer above global config that still defines preset `name`, if any — so a
 * global save can warn that the just-saved entry is shadowed and won't apply.
 */
export function modelPresetShadowOwner(
	settings: Settings,
	name: string,
): "runtime" | "overlay" | "project" | undefined {
	const owned = settings.getOwnedModelPreset(name);
	return owned && owned.source !== "global" ? owned.source : undefined;
}

const SOURCE_LABELS: Record<SettingProvenance, string> = {
	env: "environment",
	runtime: "command line",
	overlay: "--config file",
	project: "project config",
	global: "global config",
	default: "default",
};

/** Status line after saving `name`, naming the higher layer whose same-name preset still wins, if any. */
export function modelPresetSavedMessage(settings: Settings, name: string): string {
	const owner = modelPresetShadowOwner(settings, name);
	if (owner === undefined) return `Saved model preset "${name}"`;
	return `Saved model preset "${name}" to the global config, but the ${SOURCE_LABELS[owner]} still defines a preset of the same name, which takes precedence`;
}

/** Serialize default-role mutations with the model hub's assign/unassign paths. */
let modelRoleMutationTail: Promise<void> = Promise.resolve();
export async function acquireModelRoleMutation(): Promise<() => void> {
	const previous = modelRoleMutationTail;
	const { promise, resolve } = Promise.withResolvers<void>();
	modelRoleMutationTail = previous.then(() => promise);
	await previous;
	return resolve;
}

/** A role whose effective assignment differs from the preset after applying it. */
export interface ModelPresetShadowedRole {
	role: string;
	/** The preset's selector, or undefined when the preset leaves the role unset. */
	expected: string | undefined;
	/** The effective selector that won instead. */
	actual: string | undefined;
	source: SettingProvenance;
}

/** The preset's `defaultThinkingLevel` when a layer above global config still decides the setting. */
export interface ModelPresetShadowedThinking {
	expected: NonNullable<ModelPreset["defaultThinkingLevel"]>;
	/** The effective setting that won instead; the live session still uses `expected`. */
	actual: ModelPreset["defaultThinkingLevel"];
	source: SettingProvenance;
}

export type ModelPresetSwitchResult =
	| {
			kind: "switched";
			model: Model;
			thinkingLevel: ConfiguredThinkingLevel | undefined;
			/** Roles another layer still decides; empty on a clean switch. */
			shadowed: ModelPresetShadowedRole[];
			/** Set when another layer still decides `defaultThinkingLevel`, so it returns on reload. */
			shadowedThinking: ModelPresetShadowedThinking | undefined;
	  }
	| { kind: "missing" }
	| { kind: "invalid"; reason: string }
	/** Nothing was written: the preset's default model cannot be used right now. */
	| { kind: "unavailable"; reason: string }
	/** Roles were written but the live model could not be switched. */
	| { kind: "failed"; reason: string; shadowed: ModelPresetShadowedRole[] };

export type ModelPresetSession = Pick<
	AgentSession,
	"setModel" | "setThinkingLevel" | "getAvailableModels" | "scopedModels" | "modelRegistry" | "setActivePreset"
>;

function presetCandidates(session: ModelPresetSession): Model[] {
	const scoped = session.scopedModels.map(entry => entry.model);
	return scoped.length > 0 ? scoped : session.getAvailableModels();
}

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

/** First authed model for a preset with no `default`, mirroring startup's automatic pick. */
function pickAutomaticDefault(session: ModelPresetSession, candidates: Model[]): Model | undefined {
	return pickDefaultAvailableModel(
		candidates.filter(candidate => session.modelRegistry.hasConfiguredAuth(candidate)),
		provider => session.modelRegistry.hasConcreteAuth(provider),
	);
}

/**
 * Where `default` lands once the preset is written: its model and thinking selector.
 * Without an explicit `:level` on the selector, the preset's own `defaultThinkingLevel`
 * applies even when a higher layer still decides the setting — the switch is for this
 * session; the shadowing layer is reported separately.
 */
function resolveLiveDefault(
	settings: Settings,
	session: ModelPresetSession,
	candidates: Model[],
	preset: ModelPreset,
): { model: Model; thinkingLevel: ConfiguredThinkingLevel | undefined } | string {
	const roleValue = settings.getModelRole("default");
	const fallbackLevel =
		preset.defaultThinkingLevel ?? parseConfiguredThinkingLevel(cfgDefaultThinkingLevel.get(settings));
	if (!roleValue) {
		const model = pickAutomaticDefault(session, candidates);
		return model ? { model, thinkingLevel: fallbackLevel } : "no model with configured credentials is available";
	}
	const resolved = resolveModelRoleValue(roleValue, candidates, { settings });
	if (!resolved.model) return `default model \`${roleValue}\` is not available`;
	if (!session.modelRegistry.hasConfiguredAuth(resolved.model)) {
		return `no credentials for ${resolved.model.provider}/${resolved.model.id}`;
	}
	return {
		model: resolved.model,
		thinkingLevel: resolved.explicitThinkingLevel ? resolved.thinkingLevel : fallbackLevel,
	};
}

function shadowedRoles(settings: Settings, preset: ModelPreset): ModelPresetShadowedRole[] {
	const effective = settings.getModelRoles();
	const roles = new Set([...Object.keys(effective), ...Object.keys(preset.modelRoles)]);
	const shadowed: ModelPresetShadowedRole[] = [];
	for (const role of [...roles].sort((a, b) => a.localeCompare(b))) {
		const expected = Object.hasOwn(preset.modelRoles, role) ? preset.modelRoles[role] : undefined;
		const actual = Object.hasOwn(effective, role) ? effective[role] : undefined;
		if (expected !== actual) {
			shadowed.push({ role, expected, actual, source: settings.getModelRoleProvenance(role) });
		}
	}
	return shadowed;
}

function shadowedThinking(settings: Settings, preset: ModelPreset): ModelPresetShadowedThinking | undefined {
	const expected = preset.defaultThinkingLevel;
	if (expected === undefined) return undefined;
	const actual = cfgDefaultThinkingLevel.get(settings);
	if (actual === expected) return undefined;
	return { expected, actual, source: cfgDefaultThinkingLevel.provenance(settings) };
}

/**
 * Apply preset `name`: switch the live session to the preset's default model and
 * thinking level, then record the preset on the session. The shared config is
 * never written — the active preset is per-session state, so concurrent sessions
 * on different presets do not clobber each other.
 *
 * The preset's own default is validated before the live switch, so a preset whose
 * model is gone is refused as `unavailable`. A live switch that still fails
 * afterwards is reported as `failed` and never records the preset, so the prior
 * `activePreset` is retained.
 */
export async function applyModelPreset(
	settings: Settings,
	session: ModelPresetSession,
	name: string,
): Promise<ModelPresetSwitchResult> {
	const release = await acquireModelRoleMutation();
	try {
		return await applyModelPresetLocked(settings, session, name);
	} finally {
		release();
	}
}

async function applyModelPresetLocked(
	settings: Settings,
	session: ModelPresetSession,
	name: string,
): Promise<ModelPresetSwitchResult> {
	const lookup = getModelPreset(settings, name);
	if (lookup.kind !== "found") return lookup;
	const { preset } = lookup;
	const candidates = presetCandidates(session);

	const presetDefault = Object.hasOwn(preset.modelRoles, "default") ? preset.modelRoles.default : undefined;
	if (presetDefault) {
		// Resolve against the preset's own roles overlaid on the current ones:
		// a default written as an alias (`@slow`) must see the state the write
		// produces, not the current roles.
		const roleLookup = {
			getModelRole: (role: string) =>
				Object.hasOwn(preset.modelRoles, role) ? preset.modelRoles[role] : settings.getModelRole(role),
		};
		const resolved = resolveModelRoleValue(presetDefault, candidates, { settings, roleLookup });
		if (!resolved.model)
			return { kind: "unavailable", reason: `default model \`${presetDefault}\` is not available` };
		if (!session.modelRegistry.hasConfiguredAuth(resolved.model)) {
			return {
				kind: "unavailable",
				reason: `no credentials for ${resolved.model.provider}/${resolved.model.id}`,
			};
		}
	} else if (!pickAutomaticDefault(session, candidates)) {
		// No `default` to apply and nothing to fall back to: refuse before writing.
		return { kind: "unavailable", reason: "no model with configured credentials is available" };
	}

	// The session's effective settings: the base overlaid with the preset's roles.
	// The shared config is never written; a live switch that fails below retains
	// the prior preset because the journal lands only after it succeeds.
	const overlay = buildPresetOverlay(settings, name, roleOverridesAboveBase(settings));
	const shadowed = shadowedRoles(settings, preset);
	const thinkingShadow = shadowedThinking(settings, preset);

	const live = resolveLiveDefault(overlay, session, candidates, preset);
	if (typeof live === "string") return { kind: "failed", reason: live, shadowed };
	try {
		await session.setModel(live.model, "default", { persist: false });
		// setModel re-applies the model's default or keeps the current level; the preset decides instead.
		// `:inherit` means "no explicit level": leave what setModel applied rather than clearing the session level.
		if (live.thinkingLevel !== undefined && live.thinkingLevel !== ThinkingLevel.Inherit) {
			session.setThinkingLevel(live.thinkingLevel);
		}
		// Only a completed live switch records the preset on the session; a failed
		// setModel above leaves the prior activePreset intact.
		await session.setActivePreset(name);
	} catch (error) {
		return { kind: "failed", reason: errorMessage(error), shadowed };
	}
	return {
		kind: "switched",
		model: live.model,
		thinkingLevel: live.thinkingLevel,
		shadowed,
		shadowedThinking: thinkingShadow,
	};
}

function describeRoleValue(value: string | undefined): string {
	return value ?? "(unset)";
}

/** One line per role (and the thinking level) the preset could not set, for status output. */
export function describeShadowedRoles(
	shadowed: readonly ModelPresetShadowedRole[],
	thinking?: ModelPresetShadowedThinking,
): string[] {
	const lines = shadowed.map(
		entry =>
			`${entry.role}: ${describeRoleValue(entry.actual)} from ${SOURCE_LABELS[entry.source]} (preset: ${describeRoleValue(entry.expected)})`,
	);
	if (thinking) {
		lines.push(
			`defaultThinkingLevel: ${describeRoleValue(thinking.actual)} from ${SOURCE_LABELS[thinking.source]} (preset: ${thinking.expected})`,
		);
	}
	return lines;
}

/** True when the switch applied the whole preset: switched, with no role or thinking level still decided elsewhere. */
export function isCleanModelPresetSwitch(result: ModelPresetSwitchResult): boolean {
	return result.kind === "switched" && result.shadowed.length === 0 && result.shadowedThinking === undefined;
}

/** Short summary of a switch outcome for status lines and command output. */
export function formatModelPresetSwitch(name: string, result: ModelPresetSwitchResult): string {
	switch (result.kind) {
		case "missing":
			return `Preset not found: ${name}`;
		case "invalid":
			return `Preset "${name}" is malformed: ${result.reason}`;
		case "unavailable":
			return `Preset "${name}" not applied: ${result.reason}`;
		case "failed":
			return `Preset "${name}" roles saved, but the model was not switched: ${result.reason}`;
		case "switched": {
			const thinking = result.thinkingLevel ? ` · ${result.thinkingLevel}` : "";
			const model = `${result.model.provider}/${result.model.id}${thinking}`;
			const elsewhere = describeShadowedRoles(result.shadowed, result.shadowedThinking);
			if (elsewhere.length === 0) return `Switched to preset "${name}" (${model})`;
			return `Switched to preset "${name}" (${model}); still set elsewhere: ${elsewhere.join("; ")}`;
		}
	}
}
