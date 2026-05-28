"""Unit tests for !TODO YAML sentinel."""
import pytest
import yaml

from mlops.todo_yaml import TAGS, TodoSentinel, register


@pytest.fixture(autouse=True)
def _register():
    register()


def test_construct_does_not_raise():
    s = TodoSentinel("unit2.lab1 — choose dim")
    assert s.tag == "unit2.lab1 — choose dim"


def test_repr_does_not_raise():
    s = TodoSentinel("pick a value")
    r = repr(s)
    assert "TodoSentinel" in r
    assert "pick a value" in r


@pytest.mark.parametrize(
    "coerce",
    [int, float, bool, lambda x: x.__index__(), str],
    ids=["int", "float", "bool", "index", "str"],
)
def test_coercion_raises_with_tag(coerce):
    tag = "unit2.lab1 — choose dim"
    s = TodoSentinel(tag)
    with pytest.raises(NotImplementedError) as exc_info:
        coerce(s)
    assert tag in str(exc_info.value)


@pytest.mark.parametrize("tag", TAGS, ids=lambda t: t.lstrip("!"))
def test_yaml_load_scalar(tag):
    """Both !TODO and !TODO-LAB produce a TodoSentinel."""
    data = yaml.safe_load(f'{tag} "pick a value"')
    assert isinstance(data, TodoSentinel)
    assert data.tag == "pick a value"


def test_yaml_load_nested():
    src = """
    model:
      dim: !TODO "TODO-LAB unit1.lab2"
      lr: 3.0e-4
    """
    data = yaml.safe_load(src)
    assert isinstance(data["model"]["dim"], TodoSentinel)
    assert data["model"]["dim"].tag == "TODO-LAB unit1.lab2"
    assert data["model"]["lr"] == 3.0e-4


def test_yaml_load_lab_form():
    """!TODO-LAB is the longer-form alias; the tag content becomes .tag."""
    data = yaml.safe_load('!TODO-LAB "unit1.lab2"')
    assert isinstance(data, TodoSentinel)
    assert data.tag == "unit1.lab2"


def test_yaml_lowercase_todo_is_unrecognized():
    """The tag is case-sensitive; lowercase `!todo` should NOT be recognized
    (it would mask a misspelling). PyYAML raises a ConstructorError."""
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load('!todo "should fail"')


def test_register_is_idempotent():
    register()
    register()
    data = yaml.safe_load('!TODO "x"')
    assert isinstance(data, TodoSentinel)


def test_yaml_load_with_csafeloader():
    """jsonargparse's DefaultLoader subclasses yaml.CSafeLoader (the
    libyaml-backed C loader) whenever libyaml is installed. SafeLoader and
    CSafeLoader are *sibling* classes — registering on one does not reach
    the other — so this is the integration that broke when register() only
    touched SafeLoader.
    """
    if not hasattr(yaml, "CSafeLoader"):
        pytest.skip("libyaml CSafeLoader not available in this Python install")
    data = yaml.load('!TODO "pick a value"', Loader=yaml.CSafeLoader)
    assert isinstance(data, TodoSentinel)
    assert data.tag == "pick a value"


def test_yaml_load_via_jsonargparse_default_loader():
    """End-to-end check that jsonargparse's own DefaultLoader parses !TODO
    after register(). If this fails, the cascade chokes at parse time and
    LightningCLI never instantiates the model. The integration relies on
    CSafeLoader inheritance, so this is the test that would have caught
    the original bug.
    """
    try:
        from jsonargparse._loaders_dumpers import get_yaml_default_loader
    except ImportError:
        pytest.skip("jsonargparse layout not recognized")
    loader = get_yaml_default_loader()
    data = yaml.load('!TODO-LAB "lab-tag"', Loader=loader)
    assert isinstance(data, TodoSentinel)
    assert data.tag == "lab-tag"
