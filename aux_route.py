"""Auxiliary-route wiring for the System One provider plugin.

Hermes' auxiliary resolver builds a plain OpenAI client for a named custom
provider and hands it to ``_wrap_transport``, which recognises only the in-tree
dialects (codex, anthropic). A plugin dialect registered through the public
``register_transport`` seam is not consulted there, so an ``api_mode: systemone``
auxiliary route posts to ``/v1/chat/completions`` and 404s.

Two gaps produce that:

1. ``_resolve_named_custom_branch`` reads the entry's ``api_mode`` into
   ``entry_api_mode`` but never puts it on the request, so ``_wrap_transport``
   sees ``req.api_mode is None``.
2. ``_wrap_transport`` does not ask ``registered_api_modes()``, so even a
   populated ``req.api_mode`` would fall through to the OpenAI wire.

This module closes both from the plugin side, without editing Hermes: it wraps
the two functions and restores the in-tree behaviour for every route that is not
this plugin's dialect. ``agent/transports/__init__.py`` states the intended
contract — a registered transport is a public seam and callers that validate an
api_mode should consult ``registered_api_modes()`` — so the wrapper only supplies
what the auxiliary path omits.

Installation timing: plugin discovery imports this package while
``agent.auxiliary_client`` may still be mid-import, where the module object
exists but its definitions do not. Touching an attribute then raises, and the
loader reports "failed to load plugin". So ``install()`` is called from two
places — once at import (best effort, silently skipped when the host module is
not ready) and once from ``SystemOneClient.__init__``, which by construction
runs only after every host module is loaded. The wrappers are idempotent.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: In-tree dialects. ``_wrap_transport`` already handles these; the wrapper must
#: not intercept them or it would replace working behaviour with a no-op.
_IN_TREE_MODES = frozenset(
    {"chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse", "codex_app_server"}
)

_MARKER = "_hermes_systemone_aux_patch"


def _entry_api_mode(provider: str) -> str:
    """The ``api_mode`` a named custom provider entry declares, or ``""``.

    Reads through the same accessor the auxiliary resolver uses, so a
    ``custom_providers`` list entry and a ``providers:`` mapping entry both
    resolve. A lookup failure is not fatal: the route simply keeps the
    behaviour it had before this patch.
    """
    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider

        entry = _get_named_custom_provider(provider)
    except Exception:
        return ""
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("api_mode") or entry.get("transport") or "").strip().lower()


def install() -> bool:
    """Wrap the auxiliary resolver's two seams. Returns True when installed.

    Callable more than once; the second call is a no-op. Returns False when
    Hermes' auxiliary client is not importable (docs build, standalone tests) or
    is still mid-import.
    """
    try:
        from agent import auxiliary_client as ac
        from agent.transports import registered_api_modes
    except Exception:
        return False
    if getattr(ac, _MARKER, False):
        return True
    if not hasattr(ac, "_resolve_named_custom_branch") or not hasattr(ac, "_wrap_transport"):
        # Mid-import: the module object exists but its definitions do not yet.
        return False

    original_branch = ac._resolve_named_custom_branch
    original_wrap = ac._wrap_transport

    def branch_with_api_mode(req):
        """Put the entry's ``api_mode`` on the request before resolving.

        Without this, ``_wrap_transport`` receives ``req.api_mode is None`` and
        cannot tell a plugin dialect from a plain OpenAI endpoint.
        """
        if not getattr(req, "api_mode", None):
            mode = _entry_api_mode(getattr(req, "provider", "") or "")
            if mode:
                req = req._replace(api_mode=mode)
        return original_branch(req)

    def wrap_with_registered_transport(req, client_obj, final_model_str, base_url_str="", api_key_str=""):
        """Serve the plugin's own client for a registered non-in-tree dialect.

        ``profile.create_client`` is the same seam the main loop uses, so the
        auxiliary route and the main loop build the same client for the same
        entry. Anything else — including a profile that declines to build one —
        falls through to the original wrapper unchanged.
        """
        mode = str(getattr(req, "api_mode", "") or "").strip().lower()
        if mode and mode not in _IN_TREE_MODES and mode in registered_api_modes():
            try:
                from providers import get_provider_profile

                profile = get_provider_profile(getattr(req, "provider", "") or "")
                if profile is not None:
                    client = profile.create_client(api_key=api_key_str, base_url=base_url_str)
                    if client is not None:
                        logger.debug(
                            "systemone aux route: %s -> %s", mode, type(client).__name__
                        )
                        return client
            except Exception as exc:
                # A plugin-dialect route that cannot build its client must not
                # silently become an OpenAI-wire request to the same URL: that
                # 404s and reads as an approval failure. Log and fall through so
                # the caller's own error handling decides.
                logger.warning(
                    "systemone aux route: could not build a %s client (%s: %s); "
                    "falling back to the OpenAI wire",
                    mode, type(exc).__name__, exc,
                )
        return original_wrap(req, client_obj, final_model_str, base_url_str, api_key_str)

    ac._resolve_named_custom_branch = branch_with_api_mode
    ac._wrap_transport = wrap_with_registered_transport
    setattr(ac, _MARKER, True)
    logger.debug("systemone aux-route wiring installed")
    return True


# Best effort at import. Skipped silently when the host module is mid-import;
# SystemOneClient.__init__ retries once everything is loaded.
install()
