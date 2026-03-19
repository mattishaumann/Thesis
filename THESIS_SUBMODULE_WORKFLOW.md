# Thesis Submodule Workflow

This repository now contains the thesis as a Git submodule at [`tex/`](./tex).

That means there are two Git repositories:

- Main project repo: this repository
- Thesis repo: [`tex/`](./tex), which points to `https://github.com/mattishaumann/Thesis-Latex.git`

Overleaf is the source of truth for the thesis. The `Thesis-Latex` repo is the Git-backed version of that thesis, and this repo only tracks which thesis commit is currently linked at `tex/`.

## Simple Version

In simple terms:

- Your colleagues write in Overleaf.
- Overleaf syncs those changes to the `Thesis-Latex` GitHub repo.
- This repo does not update `tex/` automatically.
- When you want the newest thesis here, you update the `tex/` submodule.

So the practical rule is:

- Overleaf changes first
- then you update `tex/` here

The command for that is:

```bash
git submodule update --remote --merge tex
```

After that, your local [`tex/`](./tex) folder contains the newest thesis version from `Thesis-Latex`.

## Mental Model

- Edit code, notebooks, and project files in the main repo.
- Edit thesis files in [`tex/`](./tex).
- Commit thesis content changes in the thesis repo.
- Commit the `tex` submodule pointer in the main repo only when you want this repo to record a newer thesis version.

## Most Common Commands

Check main repo status:

```bash
git status
```

Check thesis repo status:

```bash
git -C tex status
```

See which thesis commit is checked out:

```bash
git submodule status
```

## Normal Workflow

### 1. Pull the latest thesis changes from Overleaf/GitHub

Run this from the main repo root:

```bash
git submodule update --remote --merge tex
```

This updates the local `tex/` checkout to the latest commit from the thesis repo.

Then verify:

```bash
git -C tex log --oneline -1
git status
```

If you want the main repo to remember that newer thesis version:

```bash
git add tex
git commit -m "Update thesis submodule"
```

### 2. Make thesis edits locally

Edit files inside [`tex/`](./tex), then commit inside the thesis repo:

```bash
git -C tex status
git -C tex add .
git -C tex commit -m "Edit thesis"
git -C tex push
```

After that, if you want this main repo to record the newer thesis commit:

```bash
git add tex
git commit -m "Update thesis submodule pointer"
```

### 3. Work only on code/data

Stay in the main repo root and use Git normally:

```bash
git status
git add .
git commit -m "Project changes"
```

If `tex` shows up as changed, that usually means the thesis submodule is pointing at a different commit than the one recorded in the main repo.

## How To Read Status

`git status` in the main repo means:

- did project files change?
- did the `tex` submodule pointer change?

`git -C tex status` means:

- did actual thesis files change?

This is the main distinction to remember.

## Cloning This Repo On Another Machine

Use:

```bash
git clone --recurse-submodules https://github.com/mattishaumann/Thesis.git
```

Or, after cloning:

```bash
git submodule update --init --recursive
```

## Practical Rules

- Treat Overleaf as authoritative for thesis content.
- Pull thesis updates before doing thesis-related local work.
- Commit thesis file changes in `tex/`, not only in the main repo.
- Commit the submodule pointer in the main repo when you want to pin this project to a specific thesis revision.

## Prompt To Give Codex

If you want me to update the thesis submodule for you, use this prompt:

```text
Update tex/ to the latest version from the Thesis-Latex submodule, show me what changed, and do not commit anything.
```

If you want me to also record that update in the main repo, use this:

```text
Update tex/ to the latest version from the Thesis-Latex submodule, show me what changed, and commit the updated submodule pointer in the main repo.
```

## Current Safety Backup

Before converting `tex/` into a submodule, the copied thesis folder was preserved here:

- [`tex_backup_before_submodule_20260318_1730/`](./tex_backup_before_submodule_20260318_1730/)

That backup can be deleted later once you are comfortable with the submodule setup.
