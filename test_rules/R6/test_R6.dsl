rule R6 "Deterministic Algorithm Option Not Used":
    condition:
        exists call in AST: (
            isRelevantTorchCall(call)
            and not useDeterministicPresent()
        )
    action: report "Deterministic Algorithm Option Not Used at line {lineno}"
