import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { TempDir } from "@oh-my-pi/pi-utils";
import { SkillDescriptionCatalog, type SkillDescriptionCompressor } from "../src/extensibility/skill-descriptions";
import type { Skill } from "../src/extensibility/skills";

const SKILL_DESCRIPTIONS_MODULE = path.resolve(import.meta.dir, "../src/extensibility/skill-descriptions.ts");

describe("SkillDescriptionCatalog asAuthored", () => {
	it("renders descriptions as authored and never calls the compressor", () => {
		const long = `Use when reviewing agent drift, weak tests, or prompt fixes. ${"x".repeat(120)}`;
		const skill = { name: "demo", description: long } as unknown as Skill;
		let calls = 0;
		const compress: SkillDescriptionCompressor = async () => {
			calls++;
			return "compressed";
		};
		const catalog = new SkillDescriptionCatalog({ compress, asAuthored: true });
		expect(catalog.render([skill]).at(-1)?.description).toBe(long);
		expect(catalog.snapshot([skill]).at(-1)?.description).toBe(long);
		expect(calls).toBe(0);
	});
});

describe("SkillDescriptionCatalog database lifecycle", () => {
	let tempDir: TempDir;

	beforeEach(() => {
		tempDir = TempDir.createSync("@omp-skill-desc-db-");
	});

	afterEach(async () => {
		await tempDir.remove();
	});

	// The shared store opens only outside the test runner, so the
	// open/no-open contract is probed with a child bun process per agent dir.
	async function probeCatalog(mode: "asAuthored" | "default", agentDir: string): Promise<void> {
		const script = [
			`import { SkillDescriptionCatalog } from ${JSON.stringify(SKILL_DESCRIPTIONS_MODULE)};`,
			mode === "asAuthored"
				? "new SkillDescriptionCatalog({ asAuthored: true });"
				: "new SkillDescriptionCatalog();",
		].join("\n");
		const child = Bun.spawn([process.execPath, "--eval", script], {
			cwd: path.resolve(import.meta.dir, "../.."),
			env: {
				...process.env,
				PI_CODING_AGENT_DIR: agentDir,
				OMP_PROFILE: "",
				PI_PROFILE: "",
				XDG_DATA_HOME: "",
				XDG_STATE_HOME: "",
				XDG_CACHE_HOME: "",
			},
			stdin: "ignore",
			stdout: "ignore",
			stderr: "pipe",
		});
		const [exitCode, stderr] = await Promise.all([child.exited, new Response(child.stderr).text()]);
		if (exitCode !== 0) throw new Error(`catalog probe failed: ${stderr}`);
	}

	it("opens no database for asAuthored catalogs and the shared store for default ones", async () => {
		const asAuthoredDir = path.join(tempDir.path(), "as-authored");
		const defaultDir = path.join(tempDir.path(), "default");
		await Promise.all([probeCatalog("asAuthored", asAuthoredDir), probeCatalog("default", defaultDir)]);
		expect(fs.existsSync(path.join(asAuthoredDir, "skill-descriptions.db"))).toBe(false);
		expect(fs.existsSync(path.join(defaultDir, "skill-descriptions.db"))).toBe(true);
	});
});
