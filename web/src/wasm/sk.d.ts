// Types for the Emscripten module built by scripts/build_wasm.sh.
interface SkModule {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  cwrap(name: string, ret: string, args: string[]): (...a: any[]) => any;
  ccall(name: string, ret: string, argTypes: string[], args: unknown[]): number;
  FS: { writeFile(path: string, data: Uint8Array): void; unlink(path: string): void };
}
declare function createSk(opts?: object): Promise<SkModule>;
export default createSk;
