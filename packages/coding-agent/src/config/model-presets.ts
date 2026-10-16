/**
 * Model presets: named snapshots of the role assignments (`modelRoles`) and
 * `defaultThinkingLevel`, saved and re-applied as a whole.
 *
 * Presets persist to the global config (`modelPresets.<name>`); a project config
 * may also define presets, which are listed and applied but never copied into
 * the global file. Applying a preset makes it the session's active preset: role
 * resolution runs on the fly (explicit session override → active preset → the
 * durable base), and the session's `Settings` stays the shared base, so writes
 * made through the session reach the shared config and concurrent sessions on
 * different presets do not clobber each other.
 */
import type { Model } from "@oh-my-pi/pi-ai";
import { THINKING_EFFORTS } from "@oh-my-pi/pi-catalog/effort";
import { AUTO_THINKING, type ConfiguredThinkingLevel, parseConfiguredThinkingLevel } from "@oh-my-pi/pi-tui/thinking";
import { isRecord } from "@oh-my-pi/pi-utils";
import type { AgentSession } from "../session/agent-session";
import { cfgDefaultThinkingLevel } from "../session/settings";
import { pickDefaultAvailableModel, resolveModelRoleValue } from "./model-resolver";
import { cfgModelPresets, type ModelPreset } from "./model-settings";
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
export function saveModelPreset(
	settings: Settings,
	name: string,
	effectiveRoles: Readonly<Record<string, string | undefined>> = settings.getModelRoles(),
	effectiveThinkingLevel: ConfiguredThinkingLevel | undefined = cfgDefaultThinkingLevel.get(settings),
): ModelPreset {
	if (!isValidModelPresetName(name)) {
		throw new Error(`Invalid preset name "${name}": use a letter, then letters, digits, - or _`);
	}
	const modelRoles: Record<string, string> = {};
	for (const [role, selector] of Object.entries(effectiveRoles)) {
		if (selector) modelRoles[role] = selector;
	}
	const defaultThinkingLevel =
		effectiveThinkingLevel === "inherit" || effectiveThinkingLevel === "off" ? undefined : effectiveThinkingLevel;
	const preset: ModelPreset = { modelRoles, defaultThinkingLevel };
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

const SOURCE_LABELS: Record<SettingProvenance | "session", string> = {
	env: "environment",
	runtime: "command line",
	overlay: "--config file",
	project: "project config",
	global: "global config",
	default: "default",
	session: "this session",
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

/** A role the preset cannot decide after applying. */
export interface ModelPresetShadowedRole {
	role: string;
	/** The preset's selector, or undefined when the preset leaves the role unset. */
	expected: string | undefined;
	/** The effective selector that won instead. */
	actual: string | undefined;
	/** The layer that won: an explicit session override, or the base layer for roles the preset leaves unset. */
	source: SettingProvenance | "session";
}

/** The preset's thinking default differs from the target session's effective setting. */
export interface ModelPresetShadowedThinking {
	expected: NonNullable<ModelPreset["defaultThinkingLevel"]>;
	actual: ConfiguredThinkingLevel | undefined;
	source: SettingProvenance | "session";
}

export type ModelPresetSwitchResult =
	| {
			kind: "switched";
			model: Model;
			thinkingLevel: ConfiguredThinkingLevel | undefined;
			/** Roles an explicit session override still decides; empty on a clean switch. */
			shadowed: ModelPresetShadowedRole[];
			/** Thinking default the target effective session decides; undefined on a clean switch. */
			shadowedThinking: ModelPresetShadowedThinking | undefined;
	  }
	| { kind: "missing" }
	| { kind: "invalid"; reason: string }
	/** Nothing was written: the preset's default model cannot be used right now. */
	| { kind: "unavailable"; reason: string }
	/** The preset was resolved but the live model could not be switched. */
	| {
			kind: "failed";
			reason: string;
			shadowed: ModelPresetShadowedRole[];
			shadowedThinking: ModelPresetShadowedThinking | undefined;
	  };

export type ModelPresetSession = Pick<
	AgentSession,
	| "sessionRoleOverrides"
	| "effectiveModelRoles"
	| "effectiveDefaultThinkingLevel"
	| "configuredThinkingLevel"
	| "hasExplicitThinkingOverride"
	| "getAvailableModels"
	| "scopedModels"
	| "modelRegistry"
	| "setActivePreset"
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
 * Roles the preset cannot decide after the switch: an explicit session override
 * (tier 1) still wins over the preset's role, and a role the preset leaves unset
 * keeps its base assignment.
 */
function targetShadowedRoles(
	settings: Settings,
	session: ModelPresetSession,
	preset: ModelPreset,
): ModelPresetShadowedRole[] {
	const roles = Object.keys({ ...settings.getModelRoles(), ...preset.modelRoles, ...session.sessionRoleOverrides });
	const shadowed: ModelPresetShadowedRole[] = [];
	for (const role of roles.sort((a, b) => a.localeCompare(b))) {
		const expected = Object.hasOwn(preset.modelRoles, role) ? preset.modelRoles[role] : undefined;
		const override = session.sessionRoleOverrides[role];
		const actual = override ?? expected ?? settings.getModelRole(role);
		if (expected !== actual) {
			shadowed.push({
				role,
				expected,
				actual,
				source: override !== undefined ? "session" : settings.getModelRoleProvenance(role),
			});
		}
	}
	return shadowed;
}

function targetShadowedThinking(
	settings: Settings,
	session: ModelPresetSession,
	preset: ModelPreset,
): ModelPresetShadowedThinking | undefined {
	const expected = preset.defaultThinkingLevel;
	if (expected === undefined) return undefined;
	const sessionOverride = session.hasExplicitThinkingOverride();
	const actual = sessionOverride ? session.configuredThinkingLevel() : expected;
	if (actual === expected) return undefined;
	return {
		expected,
		actual,
		source: sessionOverride ? "session" : cfgDefaultThinkingLevel.provenance(settings),
	};
}

/**
 * Apply preset `name`: resolve its default model and thinking level against the
 * durable base plus this session's explicit role overrides, switch the live
 * session to them, and record the preset as the session's active preset. The
 * shared config is never written — the active preset is per-session state, so
 * concurrent sessions on different presets do not clobber each other.
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
	// The switch target is the durable base plus this session's explicit
	// overrides only — the previous preset's roles must not leak in. An
	// explicit session override (tier 1) beats the preset's own role; a
	// default written as an alias (`@slow`) resolves against the target
	// preset's own roles.
	const roleLookup = {
		getModelRole: (role: string) =>
			session.sessionRoleOverrides[role] ??
			(Object.hasOwn(preset.modelRoles, role) ? preset.modelRoles[role] : settings.getModelRole(role)),
	};

	const shadowed = targetShadowedRoles(settings, session, preset);
	const shadowedThinking = targetShadowedThinking(settings, session, preset);
	const presetDefault =
		session.sessionRoleOverrides["default"] ??
		(Object.hasOwn(preset.modelRoles, "default") ? preset.modelRoles.default : undefined);

	let model: Model | undefined;
	let thinkingLevel: ConfiguredThinkingLevel | undefined;
	if (presetDefault) {
		const resolved = resolveModelRoleValue(presetDefault, candidates, { settings, roleLookup });
		if (!resolved.model) {
			return { kind: "unavailable", reason: `default model \`${presetDefault}\` is not available` };
		}
		if (!session.modelRegistry.hasConfiguredAuth(resolved.model)) {
			return {
				kind: "unavailable",
				reason: `no credentials for ${resolved.model.provider}/${resolved.model.id}`,
			};
		}
		model = resolved.model;
		// Without an explicit `:level` on the selector, the preset's own
		// `defaultThinkingLevel` decides.
		thinkingLevel = resolved.explicitThinkingLevel ? resolved.thinkingLevel : preset.defaultThinkingLevel;
	} else {
		// No `default` to apply: fall back to startup's automatic pick; the
		// preset's thinking level (or the base's) applies.
		model = pickAutomaticDefault(session, candidates);
		if (!model) return { kind: "unavailable", reason: "no model with configured credentials is available" };
		thinkingLevel =
			preset.defaultThinkingLevel ?? parseConfiguredThinkingLevel(cfgDefaultThinkingLevel.get(settings));
	}

	try {
		await session.setActivePreset(name, model, thinkingLevel);
	} catch (error) {
		return { kind: "failed", reason: errorMessage(error), shadowed, shadowedThinking };
	}
	return { kind: "switched", model, thinkingLevel, shadowed, shadowedThinking };
}

/** One line per role (and the thinking level) the preset could not set, for status output. */
export function describeShadowedRoles(
	shadowed: readonly ModelPresetShadowedRole[],
	thinking?: ModelPresetShadowedThinking,
): string[] {
	const lines = shadowed.map(
		entry =>
			`${entry.role}: ${entry.actual ?? "(unset)"} from ${SOURCE_LABELS[entry.source]} (preset: ${entry.expected ?? "(unset)"})`,
	);
	if (thinking) {
		lines.push(
			`defaultThinkingLevel: ${thinking.actual ?? "(unset)"} from ${SOURCE_LABELS[thinking.source]} (preset: ${thinking.expected})`,
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
			return `Preset "${name}" resolved, but the model was not switched: ${result.reason}`;
		case "switched": {
			const thinking = result.thinkingLevel ? ` · ${result.thinkingLevel}` : "";
			const model = `${result.model.provider}/${result.model.id}${thinking}`;
			const elsewhere = describeShadowedRoles(result.shadowed, result.shadowedThinking);
			if (elsewhere.length === 0) return `Switched to preset "${name}" (${model})`;
			return `Switched to preset "${name}" (${model}); still set elsewhere: ${elsewhere.join("; ")}`;
		}
	}
}
