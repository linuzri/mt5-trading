param(
    [switch]$Force,
    [switch]$SkipGit
)

$RootDir = $PSScriptRoot
$JsonlPath = Join-Path $RootDir "btcusd\logs\trades.jsonl"
$DashboardDir = Join-Path $RootDir "vercel-dashboard"
$DataDir = Join-Path $DashboardDir "data"
$TradesJsonPath = Join-Path $DataDir "trades.json"
$EmaJsonPath = Join-Path $DataDir "ema_status.json"
$CheckEmaScript = Join-Path $RootDir "btcusd\check_ema.py"

Write-Host "=== BTCUSD Dashboard Sync ===" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Ensure data dir exists
if (!(Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
}

# 1. Convert JSONL to JSON
if (Test-Path $JsonlPath) {
    try {
        $jsonObjects = @()
        Get-Content $JsonlPath | ForEach-Object {
            if ($_.Trim()) {
                $jsonObjects += ($_ | ConvertFrom-Json)
            }
        }
        $jsonArray = $jsonObjects | ConvertTo-Json -Depth 10
        $jsonArray | Out-File -FilePath $TradesJsonPath -Encoding UTF8 -NoNewline
        Write-Host "[OK] Converted JSONL -> JSON ($($jsonObjects.Count) records)" -ForegroundColor Green
    }
    catch {
        Write-Host "[FAIL] JSONL conversion: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[FAIL] JSONL not found: $JsonlPath" -ForegroundColor Red
    exit 1
}

# 2. Generate EMA status JSON
if (Test-Path $CheckEmaScript) {
    try {
        Push-Location (Join-Path $RootDir "btcusd")
        $emaOutput = python check_ema_json.py 2>&1
        if ($LASTEXITCODE -eq 0) {
            $emaOutput | Out-File -FilePath $EmaJsonPath -Encoding UTF8 -NoNewline
            Write-Host "[OK] EMA status updated" -ForegroundColor Green
        } else {
            Write-Host "[WARN] EMA script failed (market closed?): $emaOutput" -ForegroundColor Yellow
        }
        Pop-Location
    }
    catch {
        Write-Host "[WARN] EMA generation failed: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "[SKIP] check_ema.py not found" -ForegroundColor Yellow
}

# 3. Git push if changes detected
if (!$SkipGit) {
    Push-Location $RootDir
    $status = git status --porcelain -- vercel-dashboard/
    if ($status -or $Force) {
        git add vercel-dashboard/
        git commit -m "dashboard: sync data $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
        git push origin main
        Write-Host "[OK] Pushed to GitHub" -ForegroundColor Green
    } else {
        Write-Host "[SKIP] No changes to push" -ForegroundColor Yellow
    }
    Pop-Location
}

Write-Host "=== Sync Complete ===" -ForegroundColor Cyan
