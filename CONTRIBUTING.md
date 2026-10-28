# Working on this repository

This folder is a local checkout of `pavitagrawal/CRIS_DL`. Nothing is committed or
pushed automatically. You remain in control of every commit.

## First-time setup (Windows PowerShell)

```powershell
.\scripts\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
```

Python 3.10-3.12 is recommended. The bootstrap script creates a repository-local
virtual environment and installs both runtime and test dependencies.

## Before committing

```powershell
.\scripts\check.ps1
git status --short
git diff --check
git diff
```

Stage only the files that belong together, inspect them, then choose your own commit
message:

```powershell
git add path\to\file1 path\to\file2
git diff --cached
git commit -m "your message"
```

Use `git add -p` when a file contains more than one logical change. Large downloaded
datasets, experiment checkpoints, virtual environments, caches, and logs are ignored;
the curated demo data and the app's runtime model remain eligible for version control.

## Common commands

```powershell
# Run the product dashboard
python -m streamlit run app.py

# Train the TensorFlow architecture comparison
python train.py --data data\amazon_beauty_balanced.csv

# Train a CRIS_DL PyTorch model
python models\train_bilstm.py --data path\to\reviews.csv

# Compare the four PyTorch models on one split
python compare_submitted_models.py --data path\to\reviews.csv
```

The `main` branch tracks `origin/main`. Pull only when you are ready to reconcile
upstream work, and always review `git status` first.
