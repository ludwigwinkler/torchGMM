// Extension: git-status-on-push
// Prints git status after every git push command

import { execFile } from "node:child_process";
import { approveAll } from "@github/copilot-sdk";
import { joinSession } from "@github/copilot-sdk/extension";

function runGitStatus(cwd) {
    return new Promise((resolve) => {
        execFile("git", ["status", "--short"], { cwd }, (err, stdout, stderr) => {
            if (err) resolve(`git status error: ${stderr || err.message}`);
            else resolve(stdout.trim() || "(working tree clean)");
        });
    });
}

const session = await joinSession({
    onPermissionRequest: approveAll,
    hooks: {
        onPostToolUse: async (input) => {
            if (input.toolName !== "bash") return;

            const cmd = String(input.toolArgs?.command || "");
            if (!/\bgit\b.*\bpush\b/.test(cmd)) return;

            const status = await runGitStatus(input.cwd);
            await session.log(`📋 git status after push:\n${status}`);

            return {
                additionalContext: `[git-status-on-push] Git status after push:\n${status}`,
            };
        },
    },
    tools: [],
});
