# User guide

This guide is for anyone who wants to run SeqMaestro on their own data. If you are new to SeqMaestro, read the [overview](overview.md) first, then follow these pages in turn:

1. [Installation](installation.md)
2. [Configuring the language model](language-model.md)
3. [Setting up an agent](agents.md)
4. [Running with an agent](running.md)

[The example dataset](example.md) lets you see a whole run before you bring your own data. To take a single table of sequences through a run without an agent, use [the command line](command-line.md).

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Overview
:link: overview
:link-type: doc

The question SeqMaestro answers, the models and features it tries, the stages of a run and the report you receive.
:::

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Requirements, the environments the installer builds, and checking the result.
:::

:::{grid-item-card} Configuring the language model
:link: language-model
:link-type: doc

Choosing a provider, filling in `.env`, and checking that the model answers.
:::

:::{grid-item-card} Setting up an agent
:link: agents
:link-type: doc

Installing, signing in to and starting each of the four programs the agent can work in.
:::

:::{grid-item-card} Running with an agent
:link: running
:link-type: doc

The agent's work at each stage, the decisions that are yours, and monitoring a run.
:::

:::{grid-item-card} Running from the command line
:link: command-line
:link-type: doc

The `seqmaestro` command, which takes one table through a run to its report without an agent.
:::

:::{grid-item-card} The example dataset
:link: example
:link-type: doc

A run through every stage that takes minutes and needs no language model.
:::

:::{grid-item-card} Running by hand
:link: by-hand
:link-type: doc

The agent's commands in the order a run needs them.
:::

:::{grid-item-card} Troubleshooting
:link: troubleshooting
:link-type: doc

Fixes for problems with the installer, the language model, the agents or a run.
:::

::::

The [developer guide](../developer/index.md) explains what each stage computes and why. The [terminology](../glossary.md) page defines the terms these pages use.

```{toctree}
:hidden:

overview
installation
language-model
agents
running
command-line
example
by-hand
troubleshooting
```
