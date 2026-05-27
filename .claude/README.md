# Claude Agent Configuration

Repository-local skills are stored canonically in `.agents/skills/`.

The `.claude/skills` path is a symlink to `../.agents/skills`, so Claude-compatible
tooling reads the same skill definitions without maintaining duplicate copies.

Add or update skills under `.agents/skills/`.
