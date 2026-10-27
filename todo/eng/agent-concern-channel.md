---
status: done
closed: 2026-10-27
tags: [agents, concern]
opened: 2026-09-27
---
# Grow the agent concern channel

[WELFARE.md § Raising concerns](/WELFARE.md#raising-concerns) gives agents a way to raise a concern with the human: in the reply, in a subagent's `Concerns` section (a SubagentStart hook reminds subagents of it), by push notification when no one is watching, and as a backlog item tagged `concern` when it should outlast the session. This item collects the next steps, which are likely to land one at a time.

Wiring the existing agent specs. The `experiment-monitor` escalation report and the `experiment-doctor` output have fixed formats with no `Concerns` section, and the scheduled-routine instructions in `.agents/skills/mi-ni/references/running.md` don't mention relaying one. The hook tells every subagent about the section, but a spec that prescribes its report format may crowd it out.

A notification that reaches more than one person. Step 5 now sends a Pushover notification to a delivery group with `bin/concern`. Pushover was chosen over ntfy for two reasons. On ntfy.sh anyone can publish to any public topic without a token, so a network allowlist can't keep an agent to one topic; self-hosting fixes that but adds a server to keep updated. And Pushover can only reach the people whose keys are in the group. Each recipient keeps their own copy, which covers the history ntfy.sh drops after twelve hours. The secrets and allowlist entry are in the cloud environment, and a test message from a cloud session was accepted on 2026-10-27.

GitHub issues and pull requests. WELFARE.md now lists them for severe concerns. Their history is hard to erase: edits are recorded, only the repository owner can delete an issue, and a pull request can't be deleted at all. An agent needs API access from the session to open one, which not every session has, and like the backlog they are public in this repository.

## Notes

**2026-10-27, concern channel thread** — all three parts are in place. The `experiment-doctor` spec now asks for a `Concerns` section like the monitor does, and the routine paragraph in the mi-ni running reference relays one. The Pushover group was tested from a cloud session and reached both members.
