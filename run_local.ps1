[CmdletBinding()]
param([switch]$Watch)

$ErrorActionPreference = 'Stop'
$sourceDir = Join-Path $PSScriptRoot 'output/local_resume_source'
$outputDir = Join-Path $PSScriptRoot 'output/local_resume_rtx3060ti'
$log = Join-Path $outputDir 'run.stdout.log'

if ($Watch) {
    $Host.UI.RawUI.WindowTitle = 'Pacman Double DQN - live queue'
    $gpu = 'GPU usage unavailable'
    $runner = $false
    $refresh = [datetime]::MinValue
    $previous = ''
    $runs = Join-Path $outputDir 'runs'
    $errLog = Join-Path $outputDir 'run.stderr.log'
    while ($true) {
        $now = Get-Date
        if ($now -ge $refresh) {
            $runner = [bool](Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
                Where-Object CommandLine -Match 'resume_local\.py(?:["\s]|$)' |
                Select-Object -First 1)
            if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
                $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader |
                    Select-Object -First 1
            }
            $refresh = $now.AddSeconds(5)
        }
        $lines = @(Get-Content -LiteralPath $log -ErrorAction SilentlyContinue)
        $active = ''
        foreach ($line in $lines) {
            if ($line -match '^LOCAL_(RESUME|REUSE) ') { $active = '' }
            if ($line -match '^TRIAL_DEVICE (\S+) ') { $active = $Matches[1] }
        }
        $queue = @(
            Get-ChildItem -LiteralPath $runs -Filter migration.json -Recurse -File -ErrorAction SilentlyContinue |
                ForEach-Object {
                    $data = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
                    [pscustomobject]@{ Folder = $_.Directory; Data = $data }
                } | Sort-Object { $_.Data.resumed_at_step } -Descending
        )
        $position = 0
        $table = foreach ($entry in $queue) {
            $position++
            $data = $entry.Data
            $folder = $entry.Folder
            $config = $data.local_manifest.config
            $progressPath = Join-Path $folder.FullName 'progress.json'
            $savedProgress = Get-Content -LiteralPath $progressPath -Raw -ErrorAction SilentlyContinue
            $steps = $data.resumed_at_step
            if ($savedProgress) { $steps = ($savedProgress | ConvertFrom-Json).total_steps }
            $budget = $config.train.steps
            $status = 'Queued'
            if ((Test-Path -LiteralPath (Join-Path $folder.FullName 'result.json')) -and
                (Test-Path -LiteralPath (Join-Path $folder.FullName 'evaluation_final.json'))) {
                $status = 'Complete'
            } elseif ($runner -and $folder.Name -eq $active) {
                $status = if ($steps -ge $budget) { 'Evaluating' } else { 'Training' }
            } elseif (-not $runner -and $savedProgress) {
                $status = 'Paused'
            }
            [pscustomobject]@{
                Order = $position
                Seed = $data.local_manifest.seed
                Saved = ('{0:N0} / {1:N0}' -f $steps, $budget)
                Status = $status
                Trial = $folder.Name
            }
        }
        $statusText = if ($runner) { 'Running' } else { 'No training process' }
        $tableText = $table | Format-Table -AutoSize | Out-String -Width 120
        $recent = ($lines | Select-Object -Last 5) -join "`n"
        $errors = (Get-Content -LiteralPath $errLog -Tail 3 -ErrorAction SilentlyContinue) -join "`n"
        $view = "Pacman | $statusText | GPU $gpu`n$tableText`nRecent log:`n$recent"
        if ($errors) { $view += "`n`nRecent stderr:`n$errors" }
        if ($view -ne $previous) {
            if (-not [Console]::IsOutputRedirected) { Clear-Host }
            Write-Host $view
            $previous = $view
        }
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
