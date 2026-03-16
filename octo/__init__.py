"""Octo package initialization."""

from types import SimpleNamespace

import jax


# JAX removed/shifted some tree helpers across versions. Keep the older API
# surface that this repo uses so examples and scripts remain runnable.
if not hasattr(jax, "tree"):
    jax.tree = SimpleNamespace(  # type: ignore[attr-defined]
        map=jax.tree_util.tree_map,
        leaves=jax.tree_util.tree_leaves,
    )

if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map  # type: ignore[attr-defined]

if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree_util.tree_leaves  # type: ignore[attr-defined]
