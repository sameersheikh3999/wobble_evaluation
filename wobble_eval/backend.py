"""Model backends.

`agent_sdk` (default) drives Claude Code through the Claude Agent SDK, so it
authenticates with the **Claude subscription already logged in on this machine**
(`~/.claude/.credentials.json`) — no ANTHROPIC_API_KEY, no API billing.

`api` is the fallback for a machine with an API key / `ant auth login` profile.

Both expose the same call: `backend(system, user) -> str`.
"""
from __future__ import annotations

import os
import shutil
import sys
import time


def _ensure_cli_on_path():
    """The npm global bin is often missing from PATH in Git Bash / subshells."""
    if shutil.which("claude"):
        return shutil.which("claude")
    candidates = [
        os.path.join(os.environ.get("APPDATA", ""), "npm"),
        os.path.expanduser("~/.npm-global/bin"),
        "/usr/local/bin",
        os.path.expanduser("~/AppData/Roaming/npm"),
    ]
    for d in candidates:
        if d and os.path.isdir(d):
            for name in ("claude.cmd", "claude.exe", "claude"):
                p = os.path.join(d, name)
                if os.path.exists(p):
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                    return shutil.which("claude") or p
    return None


class AgentSDKBackend:
    """One-shot text completions via Claude Code (subscription auth).

    Deliberately locked down for experiment hygiene:
      * no tools            — the model cannot read files or search the web,
                              so it scores only the transcript in the prompt
      * setting_sources=None — the user's CLAUDE.md / project settings never
                              enter the scoring prompt
      * max_turns=1         — one response, no agentic loop
    """

    name = "agent_sdk"

    def __init__(self, cfg):
        self.cfg = cfg
        cli = _ensure_cli_on_path()
        if not cli:
            raise RuntimeError(
                "The `claude` CLI was not found. Install it with:\n"
                "    npm install -g @anthropic-ai/claude-code\n"
                "then re-run. (It reuses the subscription credentials already in "
                "~/.claude/.credentials.json — no API key needed.)")
        self.cli = cli
        import anyio
        import claude_agent_sdk as sdk
        self._anyio, self._sdk = anyio, sdk

        if cfg.THINKING == "disabled":
            self._thinking = sdk.ThinkingConfigDisabled(type="disabled")
        else:
            self._thinking = sdk.ThinkingConfigAdaptive(type="adaptive")

        self.calls = 0
        self.total_seconds = 0.0
        self.usage = []

    def _options(self, system):
        sdk = self._sdk
        return sdk.ClaudeAgentOptions(
            model=self.cfg.MODEL,
            system_prompt=system,
            effort=self.cfg.EFFORT,
            thinking=self._thinking,
            allowed_tools=[],
            tools=[],
            max_turns=1,
            setting_sources=None,
            permission_mode="bypassPermissions",
        )

    async def _acall(self, system, user):
        sdk = self._sdk
        parts = []
        async for msg in sdk.query(prompt=user, options=self._options(system)):
            if isinstance(msg, sdk.AssistantMessage):
                for block in msg.content:
                    if isinstance(block, sdk.TextBlock):
                        parts.append(block.text)
            elif isinstance(msg, sdk.ResultMessage):
                u = getattr(msg, "usage", None)
                if u is not None:
                    self.usage.append(u)
        return "".join(parts).strip()

    def __call__(self, system, user):
        t0 = time.time()
        try:
            out = self._anyio.run(self._acall, system, user)
        finally:
            self.calls += 1
            self.total_seconds += time.time() - t0
        return out


class APIBackend:
    """Anthropic Messages API — needs ANTHROPIC_API_KEY or an `ant auth` profile.

    NOTE: this bills the API, not the subscription. Opus 5 rejects temperature /
    top_p / top_k, so this path has no sampling dial either; effort is the knob.
    """

    name = "api"

    def __init__(self, cfg):
        self.cfg = cfg
        import anthropic
        self.client = anthropic.Anthropic()
        self.calls = 0
        self.total_seconds = 0.0
        self.usage = []

    def __call__(self, system, user):
        t0 = time.time()
        kw = dict(model=self.cfg.MODEL, max_tokens=8000, system=system,
                  messages=[{"role": "user", "content": user}],
                  output_config={"effort": self.cfg.EFFORT})
        if self.cfg.THINKING == "disabled":
            kw["thinking"] = {"type": "disabled"}
        try:
            with self.client.messages.stream(**kw) as stream:
                msg = stream.get_final_message()
        finally:
            self.calls += 1
            self.total_seconds += time.time() - t0
        self.usage.append(msg.usage)
        if getattr(msg, "stop_reason", None) == "refusal":
            return ""
        return "".join(b.text for b in msg.content if b.type == "text").strip()


def make_backend(cfg):
    if cfg.BACKEND == "agent_sdk":
        return AgentSDKBackend(cfg)
    if cfg.BACKEND == "api":
        return APIBackend(cfg)
    raise ValueError(f"unknown BACKEND {cfg.BACKEND!r} (use 'agent_sdk' or 'api')")
