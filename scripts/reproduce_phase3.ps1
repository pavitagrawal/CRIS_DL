$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath '.\.venv\Scripts\python.exe')) {
    throw "Run .\scripts\bootstrap.ps1 first."
}

$python = '.\.venv\Scripts\python.exe'
& $python prepare_multicategory_dataset.py
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed." }

& $python compare_submitted_models.py `
    --splits-dir data\amazon_50k_multicategory `
    --max-vocab 20000 `
    --max-length 200 `
    --epochs 5 `
    --out artifacts\submitted_50k_comparison
if ($LASTEXITCODE -ne 0) { throw "Four-model comparison failed." }

& $python analyze_final_results.py
if ($LASTEXITCODE -ne 0) { throw "Final analysis generation failed." }

& $python scripts\validate_project.py
if ($LASTEXITCODE -ne 0) { throw "Project validation failed." }

Write-Host "Phase 3 experiment and analysis completed."
