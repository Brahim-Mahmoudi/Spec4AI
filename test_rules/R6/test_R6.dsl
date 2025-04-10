rule R6 "Deterministic Algorithm Option Not Used":
    condition:
        exists call in AST: (
            isRelevantLibraryCall(call)
            and not (useDeterministic(call) or hasManualSeed(call))
        )
    action: report "Deterministic Algorithm Option Not Used at line {lineno}"
