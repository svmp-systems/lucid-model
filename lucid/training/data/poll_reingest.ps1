$out = "c:\Users\pranav h\Documents\GitHub\lucid-model\lucid\training\data\ai_ml_reingest_monitor.log"
$term = "C:\Users\pranav h\.cursor\projects\c-Users-pranav-h-Documents-GitHub-lucid-model\terminals\763190.txt"
$ckpt = "c:\Users\pranav h\Documents\GitHub\lucid-model\lucid\training\tree\checkpoints\saves\v.0.3-ai-ml\manifest.json"
$log = "c:\Users\pranav h\Documents\GitHub\lucid-model\lucid\training\data\ai_ml_reingest.log"
while ($true) {
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $py = Get-Process python -ErrorAction SilentlyContinue | Select-Object -First 1
  $status = if ($py) { "RUNNING pid=$($py.Id) cpu=$([math]::Round($py.CPU,0))s elapsed=$([math]::Round(((Get-Date)-$py.StartTime).TotalMinutes,1))m" } else { "NO_PYTHON" }
  $manifest = if (Test-Path $ckpt) { (Get-Item $ckpt).LastWriteTime.ToString("HH:mm:ss") } else { "missing" }
  $logBytes = if (Test-Path $log) { (Get-Item $log).Length } else { 0 }
  $exit = ""
  if (Test-Path $term) {
    $tail = Get-Content $term -Tail 3 -ErrorAction SilentlyContinue
    if ($tail -match "exit_code:") { $exit = ($tail | Select-String "exit_code").Line }
  }
  $line = "[$ts] $status | manifest=$manifest | log_bytes=$logBytes $exit"
  Add-Content -Path $out -Value $line
  if (-not $py -and $exit) { break }
  Start-Sleep -Seconds 120
}
