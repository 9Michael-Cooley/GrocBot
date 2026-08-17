"""Nothing in here is supported.

Unowned (see OWNERS). Not covered by CI. Do not import from outside this package
— several of these modules were mid-refactor when the tree was extracted and the
interfaces do not match anything else.

Should have been deleted before release. It is only still here because
tree_attention.py has the only working implementation of the mask construction
and someone will want it.
"""

import warnings

warnings.warn(
    "grokbot.experimental is unsupported and unowned; interfaces change without notice",
    stacklevel=2,
)
