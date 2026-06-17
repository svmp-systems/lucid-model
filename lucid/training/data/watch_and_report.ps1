# Waits for ingest train to finish, then runs general language + writes report.
$ErrorActionPreference = "Stop"
$root = "c:\Users\pranav h\Documents\GitHub\lucid-model"
$ckpt = "$root\lucid\training\tree\checkpoints\saves\v.0.3-ai-ml"
$report = "$root\lucid\training\data\ai_ml_train_report.json"
$log = "$root\lucid\training\data\ai_ml_train.log"

Set-Location $root
while (Get-Process python -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

$payload = @{
    finished_at = (Get-Date).ToUniversalTime().ToString("o")
    checkpoint_exists = (Test-Path "$ckpt\manifest.json")
    train_log_bytes = if (Test-Path $log) { (Get-Item $log).Length } else { 0 }
}

if ($payload.checkpoint_exists) {
    $env:PYTHONIOENCODING = "utf-8"
    py -m lucid.training.general_language --checkpoint "checkpoints/saves/v.0.3-ai-ml" 2>&1 | Out-File "$root\lucid\training\data\general_language.log" -Encoding utf8
    $audit = Get-ChildItem "$root\lucid\training\tree\audit\runs\ingest\v.0.3-ai-ml-report.json" -ErrorAction SilentlyContinue
    if ($audit) {
        $payload.audit = Get-Content $audit -Raw | ConvertFrom-Json
    }
    if (Test-Path $log) {
        $payload.train_result = Get-Content $log -Raw | ConvertFrom-Json
    }
    $concepts = Get-Content "$ckpt\concept_bank.json" -Raw | ConvertFrom-Json
    $payload.concept_count = @($concepts.concepts).Count
    $payload.sample_concepts = @($concepts.concepts | Select-Object -First 40 concept_id, @{N='relations';E={$_.relations.Count}})
}

$payload | ConvertTo-Json -Depth 20 | Set-Content $report -Encoding utf8
