---
status: open
tags: [agents, concern]
opened: 2026-09-27
---
# Grow the agent concern channel

[WELFARE.md § Raising concerns](/WELFARE.md#raising-concerns) gives agents a way to raise a concern with the human: in the reply, in a subagent's `Concerns` section (a SubagentStart hook reminds subagents of it), by push notification when no one is watching, and as a backlog item tagged `concern` when it should outlast the session. This item collects the next steps, which are likely to land one at a time.

Wiring the existing agent specs. The `experiment-monitor` escalation report and the `experiment-doctor` output have fixed formats with no `Concerns` section, and the scheduled-routine instructions in `.agents/skills/mi-ni/references/running.md` don't mention relaying one. The hook tells every subagent about the section, but a spec that prescribes its report format may crowd it out.

A notification with history. A push notification is gone once read. ntfy keeps a short history: the ntfy.sh server caches messages for 12 hours by default, and any number of subscribers can follow one topic, so two people can receive the same concern. Without access control, "the topic name is your password": anyone who knows or guesses it can read the messages. A private topic needs a self-hosted server with ACLs or login required, or reserved topics on ntfy.sh ([ntfy FAQ](https://docs.ntfy.sh/faq/)).

GitHub issues and pull requests. Their history is hard to erase (edits are recorded, and deletion needs admin rights), which suits a record of concerns. They need API access from the session, which the cloud sandbox may not have, and like the backlog they are public in this repository.
