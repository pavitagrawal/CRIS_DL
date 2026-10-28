$ErrorActionPreference = "Stop"

$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    & python --version *> $null
    if ($LASTEXITCODE -eq 0) {
        & python -m venv .venv
    } else {
        $python = $null
    }
}

if (-not $python) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & py -3 --version *> $null
        if ($LASTEXITCODE -eq 0) {
            & py -3 -m venv .venv
        } else {
            $launcher = $null
        }
    }
    if (-not $launcher) {
        throw "Python 3 was not found. Install Python 3.10-3.12 and run this script again."
    }
}

if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath '.\.venv\Scripts\python.exe')) {
    throw "The virtual environment could not be created."
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip could not be upgraded." }
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw "Project dependencies could not be installed." }
Write-Host "Ready. Activate with: .\.venv\Scripts\Activate.ps1"
