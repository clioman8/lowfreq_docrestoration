param(
  [string]$Corpus = "assets/raw/librispeech-lm-corpus.tgz",
  [string]$FontArchive = "assets/raw/liberation-fonts-ttf-2.1.5.tar.gz",
  [string]$EnglishModel = "assets/raw/eng.traineddata"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
foreach ($Asset in @($Corpus, $FontArchive, $EnglishModel)) {
  if (!(Test-Path -LiteralPath $Asset)) { throw "Missing input: $Asset. See README.md." }
}
if (!(Test-Path .venv/Scripts/python.exe)) {
  py -3 -m venv .venv
  if ($LASTEXITCODE -ne 0) { throw "Install Python 3.10 or later first." }
}
$Python = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
& $Python -m docrestore.assets fonts --archive $FontArchive --output assets/fonts
if ($LASTEXITCODE -ne 0) { throw "Font preparation failed." }
New-Item -ItemType Directory -Path assets/tessdata_best -Force | Out-Null
Copy-Item -LiteralPath $EnglishModel -Destination assets/tessdata_best/eng.traineddata -Force
& $Python -m docrestore.assets check --corpus $Corpus --eng assets/tessdata_best/eng.traineddata
if ($LASTEXITCODE -ne 0) { throw "Asset verification failed." }
& $Python -m docrestore.data --corpus $Corpus --fonts-dir assets/fonts --output data/pilot --config configs/pilot.example.json --prepare-eval
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed; inspect the error before proceeding." }
Write-Host "Pilot data ready. Run the validation-only diagnostic in README.md next."
Write-Host "Tesseract is installed separately; no long model training has been started."
