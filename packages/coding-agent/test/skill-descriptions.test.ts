import { describe, expect, it } from "bun:test";
import { SkillDescriptionCatalog, type SkillDescriptionCompressor } from "../src/extensibility/skill-descriptions";
import type { Skill } from "../src/extensibility/skills";

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
