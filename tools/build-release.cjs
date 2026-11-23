#!/usr/bin/env node
'use strict';
// Compile the theme into releases/<slug>/<version>/ with the main project's theme
// compiler. The source commit is this repository's current HEAD.
//   node tools/build-release.cjs /path/to/clean-your-tv-lg-webos
const fs = require('node:fs'),
  path = require('node:path'),
  cp = require('node:child_process');
const cytv = process.argv[2];
if (!cytv || !fs.existsSync(path.join(cytv, 'theme-api/compile.cjs')))
  throw Error('Usage: node tools/build-release.cjs /path/to/clean-your-tv-lg-webos');
const { compileSource } = require(path.resolve(cytv, 'theme-api/compile.cjs'));
const root = path.resolve(__dirname, '..'),
  read = (p) => fs.readFileSync(path.join(root, p));
const MINIMUM_HOME = '1.13.3';

(async () => {
  const commit = cp
    .execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' })
    .trim();
  const manifest = JSON.parse(read('theme.json'));
  const documents = [
    ...new Set([manifest.license.textPath, manifest.license.creditsPath])
  ].map((name) => ({ path: name, data: read(name) }));
  const result = await compileSource({
    manifest,
    css: read('theme.css').toString('utf8'),
    assets: manifest.assets.map((a) => Object.assign({}, a, { data: read(a.path) })),
    documents,
    videos: [],
    source: { repository: 'https://github.com/' + manifest.id.split(':')[1], commit },
    minimumHomeVersion: MINIMUM_HOME
  });
  const m = result.manifest,
    files = new Map([['rules.json', Buffer.from(JSON.stringify(result.rules) + '\n')]]);
  m.rules.path = 'rules.json';
  for (const group of ['assets', 'documents'])
    for (const f of result[group]) files.set(f.path, f.data);
  files.set('cytv-theme.json', Buffer.from(JSON.stringify(m, null, 2) + '\n'));
  const out = path.join(root, 'releases', manifest.id.split(':')[2], m.themeVersion);
  fs.rmSync(out, { recursive: true, force: true });
  for (const [name, data] of files) {
    fs.mkdirSync(path.dirname(path.join(out, name)), { recursive: true });
    fs.writeFileSync(path.join(out, name), data);
  }
  console.log('Built ' + m.id + ' ' + m.themeVersion + ' from ' + commit);
})().catch((e) => {
  console.error(e.stack || e.message);
  process.exitCode = 1;
});
