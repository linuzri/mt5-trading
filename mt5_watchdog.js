// MT5 Watchdog — auto-restarts MetaTrader 5 if it crashes
// Sends Telegram alert on restart

const { exec, execSync } = require('child_process');
const https = require('https');

const MT5_PATH = 'C:\\Program Files\\Pepperstone MetaTrader 5\\terminal64.exe';
const BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const CHAT_ID = '3588682';
const CHECK_INTERVAL = 60_000; // 60 seconds

function sendTelegram(text) {
  if (!BOT_TOKEN) return;
  const data = JSON.stringify({ chat_id: CHAT_ID, text, parse_mode: 'HTML' });
  const req = https.request({
    hostname: 'api.telegram.org',
    path: `/bot${BOT_TOKEN}/sendMessage`,
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) }
  });
  req.on('error', () => {});
  req.write(data);
  req.end();
}

function isMT5Running() {
  try {
    const out = execSync('tasklist /FI "IMAGENAME eq terminal64.exe" /NH', { encoding: 'utf8' });
    return out.includes('terminal64.exe');
  } catch { return false; }
}

function startMT5() {
  exec(`start "" "${MT5_PATH}"`);
}

console.log(`[MT5 Watchdog] Started at ${new Date().toISOString()}`);
console.log(`[MT5 Watchdog] Monitoring: ${MT5_PATH}`);
console.log(`[MT5 Watchdog] Check interval: ${CHECK_INTERVAL / 1000}s`);

setInterval(() => {
  if (!isMT5Running()) {
    const ts = new Date().toLocaleString('en-MY', { timeZone: 'Asia/Kuala_Lumpur' });
    console.log(`[${ts}] MT5 is DOWN! Restarting...`);
    startMT5();
    
    // Check after 30s if it came back
    setTimeout(() => {
      if (isMT5Running()) {
        console.log(`MT5 restarted successfully`);
        sendTelegram(`⚠️ <b>MT5 Watchdog Alert</b>\n\nMetaTrader 5 was found dead and has been restarted automatically.\nTime: ${ts}\n\nThe trading bot should reconnect within 1-2 minutes.`);
      } else {
        console.log(`FAILED to restart MT5!`);
        sendTelegram(`🚨 <b>MT5 Watchdog CRITICAL</b>\n\nMetaTrader 5 crashed and FAILED to restart!\nTime: ${ts}\n\nManual intervention needed.`);
      }
    }, 30_000);
  }
}, CHECK_INTERVAL);
