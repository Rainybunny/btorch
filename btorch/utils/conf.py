from collections.abc import Iterable
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


def diff_conf(
    conf_a: DictConfig | ListConfig,
    conf_b: DictConfig | ListConfig,
    mode: (Iterable[Literal["changed", "added", "removed"]] | None) = None,
) -> DictConfig | ListConfig:
    """Compare ``conf_b`` to ``conf_a`` and return a structured OmegaConf diff.

    The returned config contains only the selected changed keys. For removed
    keys (when moded by ``mode``), values are set to ``None`` so callers can
    render these entries via :func:`to_dotlist` as ``key=null``.
    """

    if not isinstance(conf_a, (DictConfig, ListConfig)):
        raise TypeError("diff_conf expects conf_a to be DictConfig or ListConfig.")
    if not isinstance(conf_b, (DictConfig, ListConfig)):
        raise TypeError("diff_conf expects conf_b to be DictConfig or ListConfig.")

    records = diff_conf_records(conf_a, conf_b, mode=mode)

    def _is_index(token: str) -> bool:
        return token.isdigit()

    def _empty_container(next_token: str):
        return [] if _is_index(next_token) else {}

    def _set_path(root, path: str, value):
        if path == "<root>":
            return value

        tokens = path.split(".")
        if root is None:
            root = _empty_container(tokens[0])

        cur = root
        for i, token in enumerate(tokens):
            is_last = i == len(tokens) - 1

            if isinstance(cur, dict):
                if is_last:
                    cur[token] = value
                    break
                if token not in cur:
                    cur[token] = _empty_container(tokens[i + 1])
                cur = cur[token]
                continue

            if isinstance(cur, list):
                idx = int(token)
                while len(cur) <= idx:
                    if is_last:
                        cur.append(None)
                    else:
                        cur.append(_empty_container(tokens[i + 1]))
                if is_last:
                    cur[idx] = value
                    break
                if cur[idx] is None:
                    cur[idx] = _empty_container(tokens[i + 1])
                cur = cur[idx]
                continue

            raise TypeError(f"Cannot set nested path '{path}' through scalar node.")

        return root

    diff_tree = None
    for path, entry in sorted(records.items()):
        if entry["status"] == "removed":
            value = None
        else:
            value = entry["new"]
        diff_tree = _set_path(diff_tree, path, value)

    if diff_tree is None:
        # Keep return type stable and flattenable.
        if isinstance(conf_b, ListConfig):
            return OmegaConf.create([])
        return OmegaConf.create({})

    if not isinstance(diff_tree, (dict, list, DictConfig, ListConfig)):
        raise ValueError(
            "diff_conf produced a scalar root diff. "
            "Expected a DictConfig/ListConfig root."
        )

    return OmegaConf.create(diff_tree)


def diff_conf_records(
    conf_a: DictConfig | ListConfig,
    conf_b: DictConfig | ListConfig,
    mode: (Iterable[Literal["changed", "added", "removed"]] | None) = None,
) -> dict[str, dict[str, object]]:
    """Compare ``conf_b`` to ``conf_a`` and return per-key value-level records.

    Each record has the shape ``{"status": str, "old": object, "new": object}``.

    - ``status='changed'``: key exists in both, value differs.
    - ``status='added'``: key exists only in ``conf_b``.
    - ``status='removed'``: key exists only in ``conf_a``.

    This representation is suitable when a caller needs both key names and values,
    for example to build child-process overrides from a baseline config.
    """

    if not isinstance(conf_a, (DictConfig, ListConfig)):
        raise TypeError(
            "diff_conf_records expects conf_a to be DictConfig or ListConfig."
        )
    if not isinstance(conf_b, (DictConfig, ListConfig)):
        raise TypeError(
            "diff_conf_records expects conf_b to be DictConfig or ListConfig."
        )

    if mode is None:
        mode_set = {"changed", "added", "removed"}
    else:
        mode_set = set(mode)

    valid_status = {"changed", "added", "removed"}
    if not mode_set.issubset(valid_status):
        raise ValueError("mode must only contain: 'changed', 'added', 'removed'.")

    changed: set[str] = set()
    added: set[str] = set()
    removed: set[str] = set()
    changed_values: dict[str, tuple[object, object]] = {}
    added_values: dict[str, object] = {}
    removed_values: dict[str, object] = {}

    def _is_container(node) -> bool:
        return isinstance(node, (DictConfig, ListConfig))

    def _collect_leaf_paths(node, path: str) -> set[str]:
        if isinstance(node, DictConfig):
            out: set[str] = set()
            for key, value in node.items():
                new_path = f"{path}.{key}" if path else str(key)
                out |= _collect_leaf_paths(value, new_path)
            return out

        if isinstance(node, ListConfig):
            out = set()
            for idx, value in enumerate(node):
                new_path = f"{path}.{idx}" if path else str(idx)
                out |= _collect_leaf_paths(value, new_path)
            return out

        return {path} if path else {"<root>"}

    def _walk(a_node, b_node, path: str = ""):
        if isinstance(a_node, DictConfig) and isinstance(b_node, DictConfig):
            keys_a = set(a_node.keys())
            keys_b = set(b_node.keys())

            for key in keys_a - keys_b:
                key_path = f"{path}.{key}" if path else str(key)
                leaf_paths = _collect_leaf_paths(a_node[key], key_path)
                removed.update(leaf_paths)
                for leaf_path in leaf_paths:
                    removed_values[leaf_path] = OmegaConf.select(conf_a, leaf_path)

            for key in keys_b - keys_a:
                key_path = f"{path}.{key}" if path else str(key)
                leaf_paths = _collect_leaf_paths(b_node[key], key_path)
                added.update(leaf_paths)
                for leaf_path in leaf_paths:
                    added_values[leaf_path] = OmegaConf.select(conf_b, leaf_path)

            for key in keys_a & keys_b:
                key_path = f"{path}.{key}" if path else str(key)
                _walk(a_node[key], b_node[key], key_path)
            return

        if isinstance(a_node, ListConfig) and isinstance(b_node, ListConfig):
            len_a = len(a_node)
            len_b = len(b_node)
            common = min(len_a, len_b)

            for idx in range(common):
                key_path = f"{path}.{idx}" if path else str(idx)
                _walk(a_node[idx], b_node[idx], key_path)

            for idx in range(common, len_a):
                key_path = f"{path}.{idx}" if path else str(idx)
                leaf_paths = _collect_leaf_paths(a_node[idx], key_path)
                removed.update(leaf_paths)
                for leaf_path in leaf_paths:
                    removed_values[leaf_path] = OmegaConf.select(conf_a, leaf_path)

            for idx in range(common, len_b):
                key_path = f"{path}.{idx}" if path else str(idx)
                leaf_paths = _collect_leaf_paths(b_node[idx], key_path)
                added.update(leaf_paths)
                for leaf_path in leaf_paths:
                    added_values[leaf_path] = OmegaConf.select(conf_b, leaf_path)
            return

        if _is_container(a_node) != _is_container(b_node):
            key = path if path else "<root>"
            changed.add(key)
            changed_values[key] = (a_node, b_node)
            return

        if a_node != b_node:
            key = path if path else "<root>"
            changed.add(key)
            changed_values[key] = (a_node, b_node)

    _walk(conf_a, conf_b)

    out: set[str] = set()
    if "changed" in mode_set:
        out |= changed
    if "added" in mode_set:
        out |= added
    if "removed" in mode_set:
        out |= removed

    records: dict[str, dict[str, object]] = {}
    for key in sorted(out):
        if key in changed:
            old_value, new_value = changed_values[key]
            records[key] = {"status": "changed", "old": old_value, "new": new_value}
            continue
        if key in added:
            records[key] = {"status": "added", "old": None, "new": added_values[key]}
            continue
        records[key] = {"status": "removed", "old": removed_values[key], "new": None}

    return records


def diff_conf_dotlist(
    conf_a: DictConfig | ListConfig,
    conf_b: DictConfig | ListConfig,
    mode: (Iterable[Literal["changed", "added", "removed"]] | None) = None,
    removed_prefix: str = "~",
) -> list[str]:
    """Build CLI-style overrides that transform ``conf_a`` into ``conf_b``.

    For ``added`` and ``changed`` entries, this emits ``"path=value"``.
    For ``removed`` entries (when moded by ``mode``), this emits
    ``"{removed_prefix}path"``. By default this produces Hydra-style remove
    overrides such as ``"~model.dropout"``.
    """

    records = diff_conf_records(conf_a, conf_b, mode=mode)

    dotlist: list[str] = []
    for key in sorted(records.keys()):
        status = records[key]["status"]
        new_value = records[key]["new"]

        if status == "removed":
            dotlist.append(f"{removed_prefix}{key}")
            continue

        if isinstance(new_value, (DictConfig, ListConfig, dict, list)):
            raise ValueError(
                "diff_conf_dotlist cannot serialize container-valued changes at "
                f"'{key}'. Use diff_conf_records for this case."
            )

        value = "null" if new_value is None else new_value
        dotlist.append(f"{key}={value}")

    return dotlist
