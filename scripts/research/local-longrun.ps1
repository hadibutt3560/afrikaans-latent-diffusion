param(
    [ValidateSet('smoke','final-test','download-scorer','score')]
    [string]$Action = 'smoke'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $repoRoot
$bundlePath = Join-Path $repoRoot 'outputs/longrun-local-inputs'
$resultPath = Join-Path $repoRoot 'outputs/longrun-local-evaluation'
$aePath = Join-Path $repoRoot 'outputs/cluster-results/afrikaans-56531/best'
$pythonPath = Join-Path $repoRoot '.venv/Scripts/python.exe'
if (!(Test-Path "$bundlePath/selection.json")) { throw 'Download and extract the cluster bundle first; see docs/research/AFRIKAANS_LONGRUN.md' }
New-Item -ItemType Directory -Force "$resultPath/logs" | Out-Null
$logPath = Join-Path $resultPath "logs/$Action-$(Get-Date -Format yyyyMMdd-HHmmss).log"
# Windows PowerShell turns native stderr (including harmless progress bars) into
# ErrorRecords. Let the process finish and use its exit code to detect failure.
$previousPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $pythonPath -u tools/local_longrun_evaluation.py $Action --bundle $bundlePath --output $resultPath --ae $aePath 2>&1 |
        ForEach-Object { $_.ToString() } | Tee-Object -FilePath $logPath
    $evaluationExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
}
if ($evaluationExitCode -ne 0) { throw "Evaluation exited with code $evaluationExitCode. See $logPath" }
