import { requireOptionalNativeModule } from 'expo';

export interface SkSolverNative {
  load(round: number, path: string): Promise<boolean>;
  query(text: string): string;
}

// null on platforms without the native solver (e.g. Android for now).
export default requireOptionalNativeModule<SkSolverNative>('SkSolver');
