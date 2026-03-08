const { spawn } = require('child_process');
const args = ['autotrader.py', '--hours', '168', '--delay', '120'];
const proc = spawn('python', args, { stdio: 'inherit', shell: true, cwd: __dirname });
proc.on('exit', code => process.exit(code));
