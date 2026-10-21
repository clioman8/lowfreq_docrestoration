$ErrorActionPreference = 'Stop'
$studyWorkspace = Split-Path -Parent $PSScriptRoot
$studyPython = Join-Path $studyWorkspace '.venv\Scripts\python.exe'
$studyRunner = Join-Path $PSScriptRoot 'run_study.py'
$studyLogDir = Join-Path $studyWorkspace 'experiment\logs'
$studyStatusPath = Join-Path $studyWorkspace 'experiment\study_status.json'
$studyPrevious = Get-Content -LiteralPath $studyStatusPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($studyExistingPid in @($studyPrevious.parent_pid, $studyPrevious.child_pid)) {
    if ($studyExistingPid -and (Get-Process -Id $studyExistingPid -ErrorAction SilentlyContinue)) {
        throw "Existing reported study process $studyExistingPid is still alive; do not duplicate the pipeline."
    }
}
New-Item -ItemType Directory -Path $studyLogDir -Force | Out-Null
$studyStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$studyOutLog = Join-Path $studyLogDir ("pipeline_background_" + $studyStamp + ".out.log")
$studyErrLog = Join-Path $studyLogDir ("pipeline_background_" + $studyStamp + ".err.log")
$studyStarted = Start-Process -FilePath $studyPython -ArgumentList @('-u', ('"' + $studyRunner + '"')) -WorkingDirectory $studyWorkspace -RedirectStandardOutput $studyOutLog -RedirectStandardError $studyErrLog -WindowStyle Hidden -PassThru
[ordered]@{ launcher_pid=$studyStarted.Id; stdout=$studyOutLog; stderr=$studyErrLog; started_at=(Get-Date).ToString('o') } | ConvertTo-Json
