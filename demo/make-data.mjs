// Regenerate data.json from the demo source of truth.
// Usage: node make-data.mjs
import fs from 'node:fs';

const src = new URL('../delegation-walkthrough/src/demoData.ts', import.meta.url);
const dst = new URL('./data.json', import.meta.url);
const text = fs.readFileSync(src, 'utf8');
const start = text.indexOf('{', text.indexOf('DemoData ='));
const end = text.lastIndexOf('}');
if (start < 0 || end < 0) throw new Error('demoData.ts structure changed');
const data = JSON.parse(text.slice(start, end + 1).replace(/,\s*([}\]])/g, '$1'));
fs.writeFileSync(dst, JSON.stringify(data, null, 1));
console.log('wrote', dst.pathname, fs.statSync(dst).size, 'bytes');
