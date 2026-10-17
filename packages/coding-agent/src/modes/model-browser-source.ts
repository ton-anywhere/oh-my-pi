import { type Model, resolveModelServiceTier, type ServiceTier, shouldSendServiceTier } from "@oh-my-pi/pi-ai";
import type { ModelHubSource } from "@oh-my-pi/pi-tui/overlays/model-hub";
import { findActiveModelPreset, getModelPreset, getModelPresetNames } from "../config/model-presets";
import { resolveModelRoleValue, rolePriorityDefaults } from "../config/model-resolver";
import { getEffectiveKnownRoleIds, getRoleInfo } from "../config/model-roles";
import { buildServiceTierByFamily } from "../config/service-tier";
import type { Settings } from "../config/settings";
import { describeModelCompactionPoint } from "../session/model-compaction-threshold";

import {
	cfgCycleOrder,
	cfgDisabledProviders,
	cfgModelProviderOrder,
	cfgModelRoleStorage,
} from "../config/model-settings";
import {
	cfgDefaultThinkingLevel,
	cfgRetryFallbackChains,
	cfgTierAnthropic,
	cfgTierGoogle,
	cfgTierOpenai,
} from "../session/settings";
import type { ConfiguredThinkingLevel } from "@oh-my-pi/pi-tui/thinking";

/**
 * Supply live model-overlay preferences and runtime resolution from the host.
 * @param settings - Settings backing preferences, roles, and model perf
 * @param sessionServiceTier - The live session's effective tier for a model
 *   (`/fast`, `/fast ultra`, `/slow`, resumed tiers). Omit only where no session
 *   exists; the configured `tier.*` settings stand in then.
 * @param activePreset - The live session's active preset, for the hub's active
 *   marker. Omit to fall back to `findActiveModelPreset(settings)`.
 * @param effectiveModelRole - The live session's effective role selectors
 *   (explicit override → active preset → base). Omit only where no session
 *   exists; the base `modelRoles` stand in then.
 */
export function createModelBrowserSource(
	settings: Settings,
	sessionServiceTier?: (model: Model) => ServiceTier | undefined,
	activePreset?: () => string | undefined,
	effectiveModelRole?: (role: string) => string | undefined,
	effectiveDefaultThinkingLevel?: () => ConfiguredThinkingLevel | undefined,
	sessionRoleOverrides?: () => Readonly<Record<string, string>>,
): ModelHubSource {
	return {
		get revision() {
			return settings.revision;
		},
		get defaultThinkingLevel() {
			return effectiveDefaultThinkingLevel?.() ?? cfgDefaultThinkingLevel.get(settings);
		},
		get modelProviderOrder() {
			return cfgModelProviderOrder.get(settings);
		},
		get knownRoleIds() {
			const presetName = activePreset?.();
			const preset = presetName ? getModelPreset(settings, presetName) : undefined;
			const presetRoleKeys = preset?.kind === "found" ? Object.keys(preset.preset.modelRoles) : [];
			return getEffectiveKnownRoleIds(settings, presetRoleKeys, Object.keys(sessionRoleOverrides?.() ?? {}));
		},
		get mruOrder() {
			return settings.getStorage()?.getModelUsageOrder() ?? [];
		},
		get modelPerf() {
			return settings.getStorage()?.getModelPerf() ?? new Map();
		},
		serviceTierFor: model => {
			const tier = sessionServiceTier
				? sessionServiceTier(model)
				: resolveModelServiceTier(
						buildServiceTierByFamily(
							cfgTierOpenai.get(settings),
							cfgTierAnthropic.get(settings),
							cfgTierGoogle.get(settings),
						),
						model,
					);
			// Label only a tier the request carries as `service_tier`: that is what the
			// provider echoes back as the served tier, which keys the perf row.
			return shouldSendServiceTier(tier, model) ? tier : undefined;
		},
		get disabledProviders() {
			return cfgDisabledProviders.get(settings);
		},
		get fallbackChains() {
			return cfgRetryFallbackChains.get(settings);
		},
		get modelRoleStorage() {
			return cfgModelRoleStorage.get(settings);
		},
		get cycleOrder() {
			return cfgCycleOrder.get(settings);
		},
		getModelRole: role => effectiveModelRole?.(role) ?? settings.getModelRole(role),
		getProjectModelRole: role => settings.getProjectModelRole(role),
		getGlobalModelRole: role => settings.getGlobalModelRole(role),
		getModelRoleSource: role => settings.getModelRoleSource(role),
		getRoleInfo: role => getRoleInfo(role, settings),
		defaultRoleChain: role => rolePriorityDefaults(role),
		resolveRoleValue: (value, models, roleLookup) =>
			resolveModelRoleValue(value, models, {
				settings,
				roleLookup: roleLookup ?? (effectiveModelRole ? { getModelRole: effectiveModelRole } : undefined),
			}),
		getModelPresets: () => ({
			names: getModelPresetNames(settings),
			active: activePreset ? activePreset() : findActiveModelPreset(settings),
		}),
		get activePreset() {
			return activePreset?.();
		},
		compactionPointFor: model => describeModelCompactionPoint(settings, model),
	};
}
