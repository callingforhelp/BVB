#!/usr/bin/env node
/** Build a password-protected static page, including lazy encrypted downloads.
 * Usage: node scripts/build_private_page.cjs --source PATH --gate PATH --output PATH
 * Read the existing gate password from stdin. No dependencies beyond Node.js.
 */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const args = Object.fromEntries(process.argv.slice(2).reduce((out, value, i, all) => {
  if (i % 2 === 0) out.push([value, all[i + 1]]);
  return out;
}, []));
const source = args['--source'], gatePath = args['--gate'], output = args['--output'];
if (!source || !gatePath || !output) throw new Error('Required: --source, --gate, --output; password on stdin.');
if (fs.existsSync(output) && fs.readdirSync(output).length) throw new Error('Output must be empty.');
const password = fs.readFileSync(0, 'utf8').replace(/\r?\n$/, '');
if (!password) throw new Error('Empty password.');
const gate = fs.readFileSync(gatePath, 'utf8');
const configMatch = gate.match(/staticryptConfig\s*=\s*(\{[^\n]+\});/);
if (!configMatch) throw new Error('Expected an existing StatiCrypt gate.');
const config = JSON.parse(configMatch[1]);
const salt = config.staticryptSaltUniqueVariableName;
let hash = crypto.pbkdf2Sync(password, salt, 1000, 32, 'sha1').toString('hex');
hash = crypto.pbkdf2Sync(hash, salt, 14000, 32, 'sha256').toString('hex');
hash = crypto.pbkdf2Sync(hash, salt, 585000, 32, 'sha256').toString('hex');
const passwordKey = Buffer.from(hash, 'hex');
const original = config.staticryptEncryptedMsgUniqueVariableName;
const originalMac = crypto.createHmac('sha256', passwordKey).update(original.slice(64)).digest('hex');
if (!crypto.timingSafeEqual(Buffer.from(originalMac, 'hex'), Buffer.from(original.slice(0, 64), 'hex'))) {
  throw new Error('Password does not unlock the existing page. No output written.');
}
const key = crypto.randomBytes(32);
const privateAssets = {}, staticAssets = {};
const mime = {'.pdf':'application/pdf','.mp4':'video/mp4','.jpg':'image/jpeg','.png':'image/png','.svg':'image/svg+xml','.csv':'text/csv;charset=utf-8','.json':'application/json'};
const assets = [];
function walk(dir) {
  for (const ent of fs.readdirSync(path.join(source, dir), {withFileTypes:true})) {
    const rel = path.posix.join(dir, ent.name);
    if (ent.isDirectory()) walk(rel); else assets.push(rel);
  }
}
walk('assets');
assets.push('paper.pdf', 'results.csv', 'leaderboard.json');
fs.mkdirSync(path.join(output, 'encrypted'), {recursive:true});
for (const rel of assets) {
  const bytes = fs.readFileSync(path.join(source, rel));
  const type = mime[path.extname(rel)];
  if (!type) throw new Error(`Unsupported asset type: ${rel}`);
  if (/\.(pdf|mp4)$/.test(rel)) {
    const iv = crypto.randomBytes(12);
    const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
    const encrypted = Buffer.concat([cipher.update(bytes), cipher.final(), cipher.getAuthTag()]);
    const file = `encrypted/${crypto.randomBytes(16).toString('hex')}.bin`;
    fs.writeFileSync(path.join(output, file), encrypted);
    privateAssets[rel] = {url:file, iv:iv.toString('base64'), mime:type};
  } else {
    staticAssets[rel] = `data:${type};base64,${bytes.toString('base64')}`;
  }
}
let html = fs.readFileSync(path.join(source, 'index.html'), 'utf8');
const escapeScript = value => value.replace(/<\/script/gi, '<\\/script');
html = html.replace(/<link rel="stylesheet" href="styles.css"\s*\/>/, `<style>${fs.readFileSync(path.join(source,'styles.css'),'utf8')}</style>`);
html = html.replace(/\s*<script src="(?:data|gallery|app)\.js(?:\?[^"\s]*)?" defer><\/script>/g, '');
html = html.replace(/<video\b[\s\S]*?<\/video>/g, block => {
  const src = block.match(/<source\s+src="([^"]+)"/);
  if (!src || !privateAssets[src[1]]) throw new Error('Unrecognized initial video.');
  return block.replace('<video', `<video data-secure-src="${src[1]}"`).replace(/<source\b[^>]*\/?\s*>/g, '');
});
html = html.replace(/<a\b[^>]*>/g, tag => {
  const match = tag.match(/href="([^"]+)"/);
  if (!match || !staticAssets[match[1]]) return tag;
  const filename = path.posix.basename(match[1]);
  return tag.replace(/\sdownload(?:="[^"]*")?/, '').replace('>', ` download="${filename}">`);
});
html = html.replace(/(src|poster|href)="([^"]+)"/g, (full, attr, rel) => {
  if (staticAssets[rel]) return `${attr}="${staticAssets[rel]}"`;
  if (privateAssets[rel] && attr === 'href') return `href="#download" data-secure-download="${rel}"`;
  return full;
});
let app = fs.readFileSync(path.join(source,'app.js'),'utf8');
const oldVideo = 'v.poster = `${stem}.jpg`;\n      v.src = `${stem}.mp4`;';
if (!app.includes(oldVideo)) throw new Error('Video setup changed; update the protected adapter.');
app = app.replace(oldVideo, 'v.poster = window.BVB_STATIC_ASSETS[`${stem}.jpg`];\n      v.dataset.secureSrc = `${stem}.mp4`;\n      v.removeAttribute("src");');
const oldReady = 'function ready(video, token) {';
if (!app.includes(oldReady)) throw new Error('Video loading changed; update the protected adapter.');
app = app.replace(oldReady, `async function ready(video, token) {
    const resource = video.dataset.secureSrc;
    const url = await window.BVB_ASSET_URL(resource);
    if (token !== version || resource !== video.dataset.secureSrc) throw new Error("changed");
    if (video.src !== url) { video.src = url; video.load(); }
`);
const loader = `(() => {
  const assets = ${JSON.stringify(privateAssets)};
  window.BVB_STATIC_ASSETS = ${JSON.stringify(staticAssets)};
  const keyBytes = Uint8Array.from(atob(${JSON.stringify(key.toString('base64'))}), c => c.charCodeAt(0));
  const key = crypto.subtle.importKey('raw', keyBytes, 'AES-GCM', false, ['decrypt']);
  const cache = new Map();
  const base = location.href.split('#')[0].split('?')[0];
  window.BVB_ASSET_URL = path => {
    if (!assets[path]) return Promise.reject(new Error('Unknown protected asset.'));
    if (!cache.has(path)) cache.set(path, (async () => {
      const asset = assets[path];
      const response = await fetch(new URL(asset.url, base));
      if (!response.ok) throw new Error('Asset download failed.');
      const bytes = await crypto.subtle.decrypt({name:'AES-GCM',iv:Uint8Array.from(atob(asset.iv),c=>c.charCodeAt(0))}, await key, await response.arrayBuffer());
      return URL.createObjectURL(new Blob([bytes], {type:asset.mime}));
    })().catch(error => { cache.delete(path); throw error; }));
    return cache.get(path);
  };
  document.addEventListener('click', async event => {
    const link = event.target.closest('a[data-secure-download]');
    if (!link) return;
    event.preventDefault();
    if (link.dataset.busy) return;
    link.dataset.busy = 'true';
    const preview = link.target === '_blank' ? window.open('', '_blank') : null;
    if (preview) { preview.opener = null; preview.document.title = 'Opening BVB paper…'; }
    const oldLabel = link.textContent;
    link.textContent = 'Opening…';
    try {
      const url = await window.BVB_ASSET_URL(link.dataset.secureDownload);
      if (preview) preview.location.href = url;
      else { const a = document.createElement('a'); a.href = url; a.download = link.dataset.secureDownload.split('/').pop(); a.click(); }
    } catch (error) {
      if (preview) preview.close();
      alert('The file could not be opened. Please try again.');
    } finally { link.textContent = oldLabel; delete link.dataset.busy; }
  });
})();`;
const scripts = [loader, fs.readFileSync(path.join(source,'data.js'),'utf8'), fs.readFileSync(path.join(source,'gallery.js'),'utf8'), app];
html = html.replace('</body>', scripts.map(js => `<script>${escapeScript(js)}</script>`).join('\n') + '\n</body>');
if (/[a-z]+\d{4}[_-]conference|under review|submitted to|\/Users\/|\/home\//i.test(html)) throw new Error('Private authoring marker in source.');
const iv = crypto.randomBytes(16);
const cipher = crypto.createCipheriv('aes-256-cbc', passwordKey, iv);
const payload = iv.toString('hex') + Buffer.concat([cipher.update(html,'utf8'), cipher.final()]).toString('hex');
config.staticryptEncryptedMsgUniqueVariableName = crypto.createHmac('sha256',passwordKey).update(payload).digest('hex') + payload;
const protectedHtml = gate.replace(configMatch[1], JSON.stringify(config));
fs.writeFileSync(path.join(output,'index.html'), protectedHtml);
fs.writeFileSync(path.join(output,'.nojekyll'), '');
console.log(JSON.stringify({encryptedDownloads:Object.keys(privateAssets).length,embeddedAssets:Object.keys(staticAssets).length,gateBytes:Buffer.byteLength(protectedHtml),passwordMatchesExistingGate:true}));
