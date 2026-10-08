// Copies the C++ engine + solver sources the iOS app needs into
// modules/sk-solver/ios/cpp (CocoaPods only compiles files below the podspec).
// Runs on `npm install` (postinstall) and before every iOS build.
const fs = require('fs');
const path = require('path');

const repo = path.resolve(__dirname, '..', '..');
const dst = path.resolve(__dirname, '..', 'modules', 'sk-solver', 'ios', 'cpp');
const files = [
  'engine/src/game.cpp', 'engine/src/rules.cpp', 'engine/src/scoring.cpp',
  'solver/src/round.cpp', 'solver/src/encoding.cpp', 'solver/src/mlp.cpp',
  'solver/src/spot.cpp', 'solver/src/spot_c_api.cpp',
];
const headerDirs = ['engine/include/sk', 'solver/include/sk/solver'];

fs.rmSync(dst, { recursive: true, force: true });
const copy = (rel) => {
  fs.mkdirSync(path.dirname(path.join(dst, rel)), { recursive: true });
  fs.copyFileSync(path.join(repo, rel), path.join(dst, rel));
};
files.forEach(copy);
for (const dir of headerDirs)
  for (const f of fs.readdirSync(path.join(repo, dir)))
    if (/\.(hpp|h)$/.test(f)) copy(path.join(dir, f));
console.log(`sk-solver: copied C++ sources to ${path.relative(process.cwd(), dst)}`);
