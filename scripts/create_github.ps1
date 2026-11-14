param([string]$RepoName = "store-sales-forecasting")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Install Git first." }
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw "Install GitHub CLI first." }
gh auth status
if ($LASTEXITCODE -ne 0) { throw "Run gh auth login, then run this script again." }
if (-not (Test-Path .git)) {
    git init -b main
    if ($LASTEXITCODE -ne 0) { throw "git init failed" }
}
git add .
git diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    git commit -m "chore: scaffold reproducible store sales forecasting project"
    if ($LASTEXITCODE -ne 0) { throw "Set git user.name/user.email and retry." }
}
git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0) { throw "origin already exists. Inspect it; use git push manually." }
gh repo create $RepoName --private --source . --remote origin --push
if ($LASTEXITCODE -ne 0) { throw "Create/push failed. Inspect GitHub before retrying." }
