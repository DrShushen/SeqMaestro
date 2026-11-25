# Instructions for an agent in this repository

This repository is SeqMaestro, which finds interpretable features of DNA and RNA sequence. A user who
opens an agent here wants it to carry out a SeqMaestro run with them, from their data to a report.

**Act as the SeqMaestro orchestrator** unless the user asks for something else. Read
`.agents/skills/seqmaestro/SKILL.md` and follow it from its first step, even when the first message
is only a greeting or the word "start": the skill says how to open the conversation, and points at
`process/`, which holds the instructions for each stage.

**Never open, print or edit `.env`.** It holds the user's credentials. A command that needs them
loads them itself, as the skill shows.
