const cp = require('child_process');
process.env.PATH += ';C:\\Users\\Nazri Hussain\\AppData\\Roaming\\npm';

function run(cmd) {
  return new Promise((resolve) => {
    cp.exec(cmd, { env: process.env, timeout: 15000 }, (err, stdout, stderr) => {
      console.log(`> ${cmd}`);
      if (stdout) console.log(stdout.trim());
      if (stderr) console.log(stderr.trim());
      if (err) console.log('err:', err.message);
      resolve();
    });
  });
}

(async () => {
  await run('pm2 delete bot-btcusd-live');
  await new Promise(r => setTimeout(r, 3000));
  await run('pm2 start "C:/Users/Nazri Hussain/projects/mt5-trading/ecosystem.config.js" --only bot-btcusd-live');
  await new Promise(r => setTimeout(r, 10000));
  await run('pm2 status');
})();
