param(
    [string]$VenvPath = "venv"
)

$ErrorActionPreference = "Stop"

Write-Host "Initializing submodules..."
git submodule update --init --recursive

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment at $VenvPath..."
    python -m venv $VenvPath
}

$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    throw "Cannot find venv activation script at $activate"
}

Write-Host "Activating virtual environment..."
. $activate

Write-Host "Installing GroundingDINO (editable)..."
pip install -e GroundingDINO

Write-Host "Installing auto-annotator (editable)..."
pip install -e .

Write-Host "Done. Remember to set GROUNDING_DINO_CONFIG_PATH and GROUNDING_DINO_CHECKPOINT_PATH."
