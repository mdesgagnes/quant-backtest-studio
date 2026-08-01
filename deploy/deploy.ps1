# Publishes the project to GitHub. Windows / PowerShell.
# Requirements: Git installed (git-scm.com) and a GitHub account.
#
#   .\deploy\deploy.ps1 -User yourusername -Repo quant-backtest-studio

param(
    [Parameter(Mandatory=$true)][string]$User,
    [string]$Repo = "quant-backtest-studio",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git was not found. Install it from https://git-scm.com" -ForegroundColor Red
    exit 1
}

if (Test-Path ".streamlit\secrets.toml") {
    Write-Host "secrets.toml detected. It is ignored by .gitignore and will not be published." -ForegroundColor Yellow
}

if (-not (Test-Path ".git")) { git init -q; git branch -M $Branch }

git add -A
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "Nothing new to publish."
} else {
    git commit -q -m "Quant Backtest Studio"
}

$url = "https://github.com/$User/$Repo.git"
if (git remote | Select-String -Quiet "^origin$") {
    git remote set-url origin $url
} else {
    git remote add origin $url
}

Write-Host "Pushing to $url ..." -ForegroundColor Cyan
git push -u origin $Branch

Write-Host ""
Write-Host "Repository published. Three clicks left:" -ForegroundColor Green
Write-Host "  1. Open https://share.streamlit.io and sign in with GitHub"
Write-Host "  2. New app -> repo '$Repo', branch '$Branch', file 'app.py'"
Write-Host "  3. Deploy"
Write-Host ""
Write-Host "Password (optional): Settings -> Secrets -> password = ""..."""
