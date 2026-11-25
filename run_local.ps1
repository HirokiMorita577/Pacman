[CmdletBinding()]
param([switch]$Watch)

$ErrorActionPreference = 'Stop'
$sourceDir = Join-Path $PSScriptRoot 'output/local_resume_source'
$outputDir = Join-Path $PSScriptRoot 'output/local_resume_rtx3060ti'
$log = Join-Path $outputDir 'run.stdout.log'

if ($Watch) {
    Get-Content -LiteralPath $log -Tail 12 -Wait
    return
}

$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$resume = Join-Path $PSScriptRoot 'resume_local.py'
$plan = Join-Path $sourceDir 'plan.json'
$trials = @(
    Get-ChildItem -LiteralPath $sourceDir -Directory |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'backup_state.json') } |
        Sort-Object { (Get-Content -LiteralPath (Join-Path $_.FullName 'backup_state.json') -Raw |
            ConvertFrom-Json).total_steps } -Descending |
        Select-Object -ExpandProperty FullName
)
if ($trials.Count -eq 0) {
    throw "No source checkpoints found in $sourceDir"
}

$env:PACMAN_DEVICE = 'cuda'
$env:PACMAN_TORCH_THREADS = '1'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
& $python -u $resume --source-plan $plan --source-trial @trials --out $outputDir |
    ForEach-Object {
        Add-Content -LiteralPath $log -Value $_ -Encoding UTF8
        Write-Output $_
    }
exit $LASTEXITCODE
