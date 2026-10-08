// Renders assets/icon.svg to the app icon, splash and favicon PNGs.
//   node tools/icons.cjs   (needs `sharp`, e.g. npm i --no-save sharp)
const sharp = require('sharp');
const path = require('path');
const src = path.join(__dirname, '..', 'assets', 'icon.svg');
const out = (f) => path.join(__dirname, '..', 'assets', 'images', f);
(async () => {
  await sharp(src).resize(1024, 1024).png().toFile(out('icon.png'));
  await sharp(src).resize(512, 512).png().toFile(out('splash-icon.png'));
  await sharp(src).resize(48, 48).png().toFile(out('favicon.png'));
  console.log('icons written');
})();
