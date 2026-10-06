# Registers a Windows Task Scheduler job that runs the scheduled online
# evaluation every night (LLMOps: continuous evaluation against the live app).
#
#   powershell -ExecutionPolicy Bypass -File scripts\register_nightly_eval.ps1            # register (02:00 daily)
#   powershell -ExecutionPolicy Bypass -File scripts\register_nightly_eval.ps1 -Time 23:30
#   powershell -ExecutionPolicy Bypass -File scripts\register_nightly_eval.ps1 -Unregister
#   Start-ScheduledTask -TaskName TrafficShieldNightlyEval                                # run it now
#
# The services must be running at that time. Each run appends one line to
# evaluation/eval_history.jsonl, which Application Service exposes on /metrics
# (ts_eval_score) for the Grafana "Scheduled online evaluation" row and the
# ScheduledEvalRegression / ScheduledEvalStale alerts. Uses Gemini by default,
# so each full run spends a few cents of API quota.
param(
    [string]$Time = "02:00",
    [string]$Provider = "gemini",
    [switch]$Unregister
)

$TaskName = "TrafficShieldNightlyEval"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Log = Join-Path $Root "evaluation\scheduled_runs\nightly.log"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "Removed scheduled task $TaskName"
    return
}

if (-not (Test-Path $Python)) { throw "Python venv not found at $Python" }
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null

$Command = "`"$Python`" -m evaluation.scheduled_eval --provider $Provider --fail-on-regression >> `"$Log`" 2>&1"
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $Command" -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "Traffic Shield nightly online evaluation (evaluation/scheduled_eval.py)" -Force | Out-Null
Write-Output "Registered ${TaskName}: daily at $Time, provider=$Provider, log -> $Log"
