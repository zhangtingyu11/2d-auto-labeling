# Two-Developer Collaboration

## Branch Map

| Branch | Purpose | Primary developer |
| --- | --- | --- |
| `main` | Shared reviewed and runnable version | Both developers |
| `dev/fengyiovo` | Fengyiovo's ongoing work | Fengyiovo |
| `dev/colleague` | Colleague's ongoing work | Colleague |

All branches are visible to both collaborators. The personal branch names
prevent accidental overlap; they do not hide files.

## First-Time Setup

Each developer clones the same repository into a local folder with GitHub
Desktop. After cloning, choose the assigned personal branch from the Current
Branch menu.

## Start of Work

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
git switch dev/fengyiovo  # or dev/colleague
git merge main
git push origin HEAD
```

Resolve any merge conflict before starting new work.

## Finish a Task

```powershell
git status
git add <changed-files>
git commit -m "Describe the completed change"
git push origin HEAD
```

Open a pull request into `main`. The other developer reviews the code,
validation result, and data-safety checklist before merging.

## Shared Ownership

- `src/`: shared product code
- `configs/`: shared class and pipeline configuration
- `tests/`: shared validation
- `docs/`: shared decisions and operating instructions
- `.github/`: shared collaboration templates

Datasets, annotation exports, model weights, secrets, and generated outputs
remain outside Git. Exchange those artifacts through the agreed data storage
location and record only reproducible paths or checksums in Git.

## Conflict Rules

1. One developer owns a task while it is in progress.
2. Announce work that touches the same module before editing it.
3. Keep commits focused and push at least once per work session.
4. Never force-push `main`.
5. Revert a bad shared commit with a new commit; do not rewrite shared history.
