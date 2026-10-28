$ErrorActionPreference = "Stop"

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    & python --version *> $null
    if ($LASTEXITCODE -eq 0) {
        & python scripts\validate_project.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & python -m pytest -q
        exit $LASTEXITCODE
    }
}

$launcher = Get-Command py -ErrorAction SilentlyContinue
if ($launcher) {
    & py -3 --version *> $null
    if ($LASTEXITCODE -eq 0) {
        & py -3 scripts\validate_project.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & py -3 -m pytest -q
        exit $LASTEXITCODE
    }
}

throw "Python 3 was not found. Install Python 3.10-3.12, then run scripts/bootstrap.ps1."
