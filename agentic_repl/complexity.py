"""Program-complexity proxy for the simplify-and-reverify pass (solver.py).

Uses total AST node count, not raw character count. Two reasons, found before
implementing (see the Step 2 write-up in the project's implementation-cycle
doc):

1. Character count rewards obfuscation, not simplicity -- renaming every
   variable to a single letter shrinks it without reducing actual complexity.
   AST node count is invariant to naming and whitespace.
2. A pure DSL-primitive-call count (the other option considered, via
   agentic_repl.dsl.catalog's introspection of DSL_FUNCTIONS/DSL_CLASSES) was
   rejected as the *sole* metric: plenty of verified candidates in this
   project are hand-written pixel loops that use zero DSL primitives (see
   the README's "Nudge prompts toward DSL primitives" commit history --
   this was a known, real failure mode this project already had to fight),
   so two such candidates would always tie at 0 and the metric couldn't
   discriminate between them at all.

AST node count is standard practice for this kind of program-synthesis
Occam's-razor tie-break when a full probabilistic-grammar description length
(as in e.g. DreamCoder) isn't available -- it's what's actually being
approximated: fewer/shallower AST productions is a reasonable proxy for
fewer bits needed to encode the program. CompressARC's own MDL framing
(third_party/compress_arc) was checked and doesn't transfer here: it
measures bits to encode trained *network weights*, not source code, so
there's nothing to reuse from it for this specific comparison.
"""

from __future__ import annotations

import ast


def program_complexity(code: str) -> int:
    """Total AST node count. Lower is simpler.

    Raises SyntaxError if code doesn't parse -- callers only ever call this
    on candidates that have already executed successfully during train-pair
    verification, so a parse failure here would indicate a real bug upstream
    rather than something to paper over silently.
    """

    tree = ast.parse(code)
    return sum(1 for _ in ast.walk(tree))
