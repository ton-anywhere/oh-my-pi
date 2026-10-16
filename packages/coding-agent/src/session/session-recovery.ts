import * as fs from "node:fs";
import * as path from "node:path";
import { directoryIsEnterable, isEnoent, isEnotdir } from "@oh-my-pi/pi-utils";
import { SessionManager } from "./session-manager";
import type { SessionInfo } from "./session-listing";

export type SessionRecoveryDecision =
	| { kind: "unchanged" }
	| { kind: "relocate"; directory: string }
	| { kind: "choose-directory" };

type RepositoryMetadata = NonNullable<SessionInfo["repository"]>;
type DirectoryProbe = "usable" | "missing";

// Unlike the shared enterability helper, recovery must preserve non-missing filesystem errors.
async function probeRecordedCwd(directory: string): Promise<DirectoryProbe> {
	try {
		const stats = await fs.promises.stat(directory);
		if (!stats.isDirectory()) return "missing";
		await fs.promises.access(directory, fs.constants.X_OK);
		return "usable";
	} catch (error) {
		if (isEnoent(error) || isEnotdir(error)) return "missing";
		throw error;
	}
}

function validRepositoryMetadata(value: SessionInfo["repository"]): value is RepositoryMetadata {
	return (
		typeof value === "object" &&
		value !== null &&
		typeof value.sessionWorktreeRoot === "string" &&
		path.isAbsolute(value.sessionWorktreeRoot) &&
		typeof value.primaryWorktreeRoot === "string" &&
		path.isAbsolute(value.primaryWorktreeRoot)
	);
}

export async function resolveSessionRecovery(
	session: SessionInfo,
	currentCwd: string,
	sessionDir?: string,
): Promise<SessionRecoveryDecision> {
	if ((await probeRecordedCwd(session.cwd)) === "usable") return { kind: "unchanged" };

	if (validRepositoryMetadata(session.repository)) {
		if (await directoryIsEnterable(session.repository.primaryWorktreeRoot)) {
			return { kind: "relocate", directory: session.repository.primaryWorktreeRoot };
		}
		return { kind: "choose-directory" };
	}

	if (
		(await SessionManager.isFromRemovedWorktree(session, currentCwd, sessionDir)) &&
		(await directoryIsEnterable(currentCwd))
	) {
		return { kind: "relocate", directory: currentCwd };
	}

	return { kind: "choose-directory" };
}
