from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(
    Param,
    use_config_file=True,
    search_path=Path("."),
    argv: list[str] | None = None,
    return_cli=False,
    make_concrete: bool = True,
):
    """Doesn't support help text and Literal though."""
    defaults = OmegaConf.structured(Param)
    cli_cfg_ = cli_cfg = OmegaConf.from_cli()
    if use_config_file and "config_path" in cli_cfg:
        assert "config_path" not in Param.__dataclass_fields__
        config_path = Path(cli_cfg.config_path)
        if not config_path.is_file():
            config_path = search_path / config_path
            assert config_path.is_file()
        cfg_cli_file = OmegaConf.load(cli_cfg.config_path)
        if return_cli:
            cli_cfg_ = cli_cfg.copy()
        cli_cfg.pop("config_path")
    else:
        cfg_cli_file = OmegaConf.create()
    cfg = OmegaConf.unsafe_merge(defaults, cfg_cli_file, cli_cfg)
    if make_concrete:
        cfg = OmegaConf.to_object(cfg)

    if return_cli:
        return cfg, cli_cfg_
    return cfg


def to_dotlist(
    conf,
    use_equal: bool = True,
    include: set | None = None,
    exclude: set | None = None,
    subfield: str | None = None,
    missing_subfield_policy: Literal["raise", "empty"] = "raise",
):
    """Flatten a ``DictConfig``/``ListConfig`` into CLI-style dotlist entries.

    Parameters
    ----------
    conf:
        Root OmegaConf container. Must be ``DictConfig`` or ``ListConfig``.
    use_equal:
        If True, emit ``["a.b=1"]`` form. If False, emit ``["a.b", "1"]`` pairs.
    include, exclude:
        Optional exact-path filters applied to leaf paths.
        Paths are evaluated relative to ``subfield`` (if provided), otherwise
        relative to the root ``conf``.
    subfield:
        Optional dotted path used as the flattening start point.
        Supports list indices (e.g. ``"a.b.1"``).
    missing_subfield_policy:
        Behavior when ``subfield`` cannot be resolved.
        ``"raise"`` (default) raises ``KeyError``.
        ``"empty"`` returns ``[]``.

    Examples
    --------
    ``{"a": {"b": 1}}`` -> ``["a.b=1"]``
    ``subfield="a"`` -> ``["b=1"]``
    """

    if not isinstance(conf, (DictConfig, ListConfig)):
        raise TypeError("to_dotlist expects DictConfig or ListConfig.")

    ret = []

    def _select_subfield(cfg, path: str):
        # Traverse the dotted path against OmegaConf containers only.
        cur = cfg
        for token in path.split("."):
            if token == "":
                raise ValueError("subfield contains an empty token.")
            if isinstance(cur, DictConfig):
                if token not in cur:
                    raise KeyError(f"subfield '{path}' not found at token '{token}'.")
                cur = cur[token]
                continue
            if isinstance(cur, ListConfig):
                try:
                    idx = int(token)
                except ValueError as exc:
                    raise KeyError(
                        f"subfield '{path}' expects a list index at token '{token}'."
                    ) from exc
                if idx < 0 or idx >= len(cur):
                    raise KeyError(f"subfield '{path}' list index out of range: {idx}.")
                cur = cur[idx]
                continue
            raise KeyError(
                f"subfield '{path}' cannot descend through non-container "
                f"at token '{token}'."
            )
        return cur

    def flatten_conf(cfg, path=""):
        nonlocal ret
        # Recurse through OmegaConf containers and emit scalar leaves.
        if isinstance(cfg, DictConfig):
            items = cfg.items()
        elif isinstance(cfg, ListConfig):
            # For lists, indices become path tokens ("a.0.b").
            items = enumerate(cfg)
        else:
            # Base case: leaf scalar value.
            if path:
                # Keep "null" spelling to match OmegaConf textual conventions.
                value = "null" if cfg is None else cfg
                if include is not None:
                    if path not in include:
                        return
                if exclude is not None:
                    if path in exclude:
                        return
                if use_equal:
                    ret.append(f"{path}={value}")
                else:
                    ret += [path, str(value)]
            return

        for key, value in items:
            # For DictConfig, key is a string. For ListConfig, key is an int index.
            new_path = f"{path}.{key}" if path else str(key)

            # Recursively flatten nested configs
            if isinstance(value, (DictConfig, ListConfig)):
                flatten_conf(value, new_path)
            else:
                # Handle the final value
                flatten_conf(value, new_path)

    if subfield:
        try:
            start_cfg = _select_subfield(conf, subfield)
        except KeyError:
            if missing_subfield_policy == "empty":
                return []
            raise
    else:
        start_cfg = conf
    flatten_conf(start_cfg)
    return ret
