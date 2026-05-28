"""!TODO YAML sentinel for unfilled lab hparams.

Loading a YAML containing `!TODO "<tag string>"` (or the longer-form alias
`!TODO-LAB "<tag string>"`) produces a TodoSentinel. Any numeric/boolean/
index/string coercion of the sentinel raises NotImplementedError carrying
the tag string — surfacing the lab reference at the point of use.

Spec § 5.2.

Authoring flow (the source file the preprocessor consumes):
    dim: !TODO "unit2.lab1 — choose an embedding dimension"

Equivalent longer form (the tag itself reads as TODO-LAB; the content can
then be just the lab id):
    dim: !TODO-LAB "unit2.lab1"

Solution flow (after the preprocessor strips the !TODO branch):
    dim: 512

The preprocessor is upstream of ft4; ft4 sees only the resolved YAML. We
register both constructors unconditionally so a stray !TODO (e.g., a
student running an unprepared file) surfaces an informative error rather
than a generic YAML parse failure.

Loader registration
-------------------
PyYAML's `SafeLoader` and `CSafeLoader` are *sibling* classes — both descend
from `BaseConstructor` but neither inherits from the other. So registering
on one does not reach the other.

jsonargparse's `DefaultLoader` subclasses `yaml.CSafeLoader` whenever libyaml
is installed (which is essentially always — libyaml ships with PyYAML
wheels). Constructors added to CSafeLoader propagate to DefaultLoader
through normal Python inheritance. That's the path LightningCLI actually
exercises, and it's what makes the rest of this module work end-to-end.

We register on `yaml.SafeLoader` too, since `yaml.safe_load` (used in
several places in ft4 and most unit tests) goes through SafeLoader directly.

Case: the tags are uppercase `!TODO` and `!TODO-LAB`. YAML tags are
case-sensitive — `!todo` or `!Todo` would not match. This is consistent
with the `TODO-LAB unit*.lab*` convention used elsewhere in the course.
"""
import yaml


# The tag spellings we recognize. Both produce a TodoSentinel; the tag
# string the student supplies becomes the sentinel's `tag` attribute and
# is what shows up in the eventual NotImplementedError.
TAGS = ("!TODO", "!TODO-LAB")


class TodoSentinel:
    """Sentinel for an unfilled hparam. Any coercion raises NotImplementedError."""

    __slots__ = ("tag",)

    def __init__(self, tag: str):
        self.tag = tag

    def __repr__(self) -> str:
        return f"<TodoSentinel: {self.tag!r}>"

    def __int__(self):
        raise NotImplementedError(self.tag)

    def __float__(self):
        raise NotImplementedError(self.tag)

    def __bool__(self):
        raise NotImplementedError(self.tag)

    def __index__(self):
        raise NotImplementedError(self.tag)

    def __str__(self):
        raise NotImplementedError(self.tag)


def _construct(loader, node) -> TodoSentinel:
    return TodoSentinel(loader.construct_scalar(node))


def register() -> None:
    """Register the !TODO / !TODO-LAB constructors on PyYAML's SafeLoader and
    CSafeLoader. Idempotent."""
    loader_classes = [yaml.SafeLoader]
    c_safe_loader = getattr(yaml, "CSafeLoader", None)
    if c_safe_loader is not None:
        loader_classes.append(c_safe_loader)
    for cls in loader_classes:
        for tag in TAGS:
            cls.add_constructor(tag, _construct)
