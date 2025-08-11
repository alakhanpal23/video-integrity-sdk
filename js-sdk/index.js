const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const axios = require('axios');

// index.js: Node.js wrapper for Video Integrity SDK
//
// - embedVideo: Calls Python CLI to embed watermark
// - verifyVideo: Calls API server to verify watermark
//
// To add new features, extend these functions or add new exports.
// For new API endpoints, update verifyVideo accordingly.

function embedVideo(input, output, secret, crf = 23) {
  return new Promise((resolve, reject) => {
    const args = ['src/encoder.py', 'embed', input, output, '--crf', crf.toString()];
    if (secret) args.push('--secret', secret);
    const proc = spawn('python', args, { stdio: 'inherit' });
    proc.on('close', code => {
      if (code === 0) resolve();
      else reject(new Error('embedVideo failed'));
    });
  });
}

async function verifyVideo(filePath, secret, nFrames = 8, apiUrl = 'http://127.0.0.1:8000/verify') {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));
  form.append('secret_hex', secret);
  form.append('n_frames', nFrames);
  const resp = await axios.post(apiUrl, form, { headers: form.getHeaders() });
  return resp.data;
}

module.exports = { embedVideo, verifyVideo }; 