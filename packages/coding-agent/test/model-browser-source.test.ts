import { describe, expect, it } from "bun:test";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { cfgModelPresets } from "@oh-my-pi/pi-coding-agent/config/model-settings";
import { findActiveModelPreset } from "@oh-my-pi/pi-coding-agent/config/model-presets";
import { createModelBrowserSource } from "@oh-my-pi/pi-coding-agent/modes/model-browser-source";

describe("model browser source active marker", () => {
	it("the hub active marker follows the session preset", () => {
		const settings = Settings.isolated();
		settings.setModelRole("default", "b/d");
		// Preset "p" has a role that no longer matches the base setup.
		cfgModelPresets.setEntry(settings, "p", { modelRoles: { default: "p/d" } });

		// With a session preset active, the marker reflects it (not the stale findActiveModelPreset result).
		const withPreset = createModelBrowserSource(settings, undefined, () => "p");
		expect(withPreset.getModelPresets!().active).toBe("p");

		// Without a session preset, it falls back to findActiveModelPreset(settings).
		const noPreset = createModelBrowserSource(settings, undefined);
		expect(noPreset.getModelPresets!().active).toBe(findActiveModelPreset(settings));
	});
});
