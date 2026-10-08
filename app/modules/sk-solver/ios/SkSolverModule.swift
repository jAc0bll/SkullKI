import ExpoModulesCore

/// JS: load(round, path) -> Bool, query(text) -> JSON string. Queries take
/// well under a millisecond, so they run synchronously on the JS thread.
public final class SkSolverModule: Module {
  public func definition() -> ModuleDefinition {
    Name("SkSolver")

    AsyncFunction("load") { (round: Int, path: String) -> Bool in
      return SKSpotBridge.loadRound(Int32(round), path: path)
    }

    Function("query") { (text: String) -> String in
      return SKSpotBridge.query(text)
    }
  }
}
