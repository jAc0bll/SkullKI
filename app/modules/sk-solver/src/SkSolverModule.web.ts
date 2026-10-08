// Web: the same C++ compiled to WebAssembly (scripts/build_wasm.sh).
import type { SkSolverNative } from './SkSolverModule';

// eslint-disable-next-line @typescript-eslint/no-require-imports
const createSk = require('./web/sk-web.js') as () => Promise<{
  ccall(name: string, ret: string, types: string[], args: unknown[]): unknown;
  FS: { writeFile(path: string, data: Uint8Array): void; unlink(path: string): void };
}>;

let mod: ReturnType<typeof createSk> | null = null;
let ready: Awaited<ReturnType<typeof createSk>> | null = null;

const SkSolverWeb: SkSolverNative = {
  async load(round, url) {
    mod ??= createSk();
    const m = (ready = await mod);
    const res = await fetch(url);
    if (!res.ok) return false;
    const file = `/r${round}.bin`;
    m.FS.writeFile(file, new Uint8Array(await res.arrayBuffer()));
    const ok = m.ccall('sk_load', 'number', ['number', 'string'], [round, file]) === 1;
    m.FS.unlink(file);
    return ok;
  },
  query(text) {
    if (!ready) return JSON.stringify({ ok: false, error: 'Solver lädt noch…' });
    return ready.ccall('sk_spot', 'string', ['string'], [text]) as string;
  },
};

export default SkSolverWeb;
