from pathlib import Path

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
):
    """Converts an OmegaConf object to a list of dot-separated strings (a dot-
    list).

    e.g., {"a": {"b": 1}} becomes ["a.b=1"]
    """

    ret = []

    def flatten_conf(cfg, path=""):
        nonlocal ret
        # Determine the iterable and how to get keys/values
        if isinstance(cfg, DictConfig):
            items = cfg.items()
        elif isinstance(cfg, ListConfig):
            # For lists, use the index as the key
            items = enumerate(cfg)
        else:
            # Base case: Must be a value (int, string, float, bool, None)
            # The path must be non-empty to ensure we don't try to assign
            # the whole config object if it's a simple value at the root.
            if path:
                # OmegaConf converts None to 'null' for text formats like YAML
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

    flatten_conf(conf)
    return ret
