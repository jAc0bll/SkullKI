// Renders public/icon.svg to the PNG sizes iOS and the PWA manifest need.
import sharp from "sharp";
const src = new URL("../public/icon.svg", import.meta.url).pathname.replace(/^\/(\w:)/, "$1");
for (const [name, size] of [["apple-touch-icon.png", 180], ["icon-192.png", 192], ["icon-512.png", 512]])
  await sharp(src).resize(size, size).png().toFile(new URL(`../public/${name}`, import.meta.url).pathname.replace(/^\/(\w:)/, "$1"));
console.log("icons written");
