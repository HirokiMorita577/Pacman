[CmdletBinding()]
param([switch]$Watch)

$ErrorActionPreference = 'Stop'
$sourceDir = Join-Path $PSScriptRoot 'output/local_resume_source'
$outputDir = Join-Path $PSScriptRoot 'output/local_resume_rtx3060ti'
$log = Join-Path $outputDir 'run.stdout.log'

if ($Watch) {
    $Host.UI.RawUI.WindowTitle = 'Pacman Double DQN - live progress'
    $seen = 0
    $status = 'Waiting for checkpoint'
    $gpu = 'GPU usage unavailable'
    $gpuRefresh = [datetime]::MinValue
    while ($true) {
        if (Test-Path -LiteralPath $log) {
            $lines = @(Get-Content -LiteralPath $log)
            if ($lines.Count -lt $seen) { $seen = 0 }
            $newLines = $lines | Select-Object -Skip $seen
            foreach ($line in $newLines) {
                Write-Host $line
                if ($line -like 'TRIAL_PROGRESS*') { $status = $line }
            }
            $seen = $lines.Count
        }
        $now = Get-Date
        if ($now -ge $gpuRefresh -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
            $gpu = (nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader |
                Select-Object -First 1)
            $gpuRefresh = $now.AddSeconds(5)
        }
        Write-Progress -Activity "Pacman | $($now.ToString('HH:mm:ss')) | GPU $gpu" -Status $status
        Start-Sleep -Seconds 1
    }
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
