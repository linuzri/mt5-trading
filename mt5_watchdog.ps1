# MT5 Watchdog — auto-restarts MetaTrader 5 if it crashes
# Sends Telegram alert on restart

$MT5_PATH = "C:\Program Files\Pepperstone MetaTrader 5\terminal64.exe"
$BOT_TOKEN = $env:TELEGRAM_BOT_TOKEN
$CHAT_ID = "3588682"
$CHECK_INTERVAL = 60  # seconds

function Send-TelegramAlert($message) {
    if ($BOT_TOKEN) {
        try {
            $uri = "https://api.telegram.org/bot$BOT_TOKEN/sendMessage"
            $body = @{ chat_id = $CHAT_ID; text = $message; parse_mode = "HTML" } | ConvertTo-Json
            Invoke-RestMethod -Uri $uri -Method Post -Body $body -ContentType "application/json" -ErrorAction SilentlyContinue | Out-Null
        } catch {}
    }
}

Write-Host "[MT5 Watchdog] Started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "[MT5 Watchdog] Monitoring: $MT5_PATH"
Write-Host "[MT5 Watchdog] Check interval: ${CHECK_INTERVAL}s"

while ($true) {
    $proc = Get-Process terminal64 -ErrorAction SilentlyContinue
    if (!$proc) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host "[$timestamp] MT5 is DOWN! Restarting..."
        
        Start-Process $MT5_PATH
        Start-Sleep -Seconds 30
        
        $restarted = Get-Process terminal64 -ErrorAction SilentlyContinue
        if ($restarted) {
            $msg = "⚠️ <b>MT5 Watchdog Alert</b>`n`nMetaTrader 5 was found dead and has been restarted automatically.`nTime: $timestamp`n`nThe trading bot should reconnect within 1-2 minutes."
            Write-Host "[$timestamp] MT5 restarted successfully (PID: $($restarted.Id))"
        } else {
            $msg = "🚨 <b>MT5 Watchdog CRITICAL</b>`n`nMetaTrader 5 crashed and FAILED to restart!`nTime: $timestamp`n`nManual intervention needed."
            Write-Host "[$timestamp] FAILED to restart MT5!"
        }
        Send-TelegramAlert $msg
    }
    Start-Sleep -Seconds $CHECK_INTERVAL
}
