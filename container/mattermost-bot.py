#!/usr/bin/env python3
"""Mattermost bot bridge: forwards chat messages to a local inference server."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import ssl

import httpx
from mattermostdriver import Driver

# Fix mattermostdriver bug: its websocket code calls
# ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH) which creates a
# server-side context. We patch it to always use SERVER_AUTH (client-side).
_orig_create_default_context = ssl.create_default_context


def _patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, **kwargs):
    return _orig_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, **kwargs)


ssl.create_default_context = _patched_create_default_context


class MattermostBridge:
    """Bridge between Mattermost and a local LLM inference server."""

    def __init__(self) -> None:
        self.mm_url = os.environ.get("MATTERMOST_URL", "")
        self.mm_token = os.environ.get("MATTERMOST_TOKEN", "")
        self.mm_team = os.environ.get("MATTERMOST_TEAM", "")
        self.trigger = os.environ.get("MATTERMOST_TRIGGER", "@metalclaw")
        self.system_prompt = os.environ.get("MATTERMOST_SYSTEM_PROMPT", "")
        self.max_history = int(os.environ.get("MATTERMOST_MAX_HISTORY", "20"))
        self.ca_cert = os.environ.get("MATTERMOST_CA_CERT", "")
        self.inference_url = os.environ.get(
            "INFERENCE_URL",
            f"http://localhost:{os.environ.get('PORT', '8080')}",
        )

        if not self.mm_url or not self.mm_token:
            print("ERROR: MATTERMOST_URL and MATTERMOST_TOKEN are required", file=sys.stderr)
            sys.exit(1)

        # Strip trailing slash and protocol for the driver
        url = self.mm_url.rstrip("/")
        if url.startswith("https://"):
            url = url[len("https://"):]
        elif url.startswith("http://"):
            url = url[len("http://"):]

        scheme = "https"
        port = 443
        if self.mm_url.startswith("http://"):
            scheme = "http"
            port = 80

        # Handle explicit port in URL
        if ":" in url.split("/")[0]:
            host_part, port_str = url.split("/")[0].rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                host_part = url.split("/")[0]
            url = host_part

        # Use custom CA cert if provided, otherwise default system bundle
        verify = self.ca_cert if self.ca_cert else True

        # Ensure the websocket library also picks up the CA cert via
        # the process-level SSL environment before creating the driver
        if self.ca_cert:
            os.environ["SSL_CERT_FILE"] = self.ca_cert

        self.driver = Driver({
            "url": url,
            "token": self.mm_token,
            "scheme": scheme,
            "port": port,
            "verify": verify,
        })

        self.bot_user_id: str = ""
        self.bot_username: str = ""
        self.history: dict[str, list[dict[str, str]]] = defaultdict(list)

    def start(self) -> None:
        """Log in, resolve bot user ID, and connect WebSocket."""
        print(f"Connecting to Mattermost at {self.mm_url}", flush=True)
        self.driver.login()

        me = self.driver.users.get_user("me")
        self.bot_user_id = me["id"]
        self.bot_username = me["username"]

        # Default trigger to the bot's own @mention so it responds naturally
        if not self.trigger or self.trigger == "@metalclaw":
            self.trigger = f"@{self.bot_username}"

        print(f"Logged in as @{self.bot_username} (id={self.bot_user_id})", flush=True)
        print(f"Trigger: {self.trigger}", flush=True)
        print(f"Inference: {self.inference_url}/v1", flush=True)

        self.driver.init_websocket(self._on_event)

    async def _on_event(self, event: str) -> None:
        """WebSocket event dispatcher."""
        try:
            msg = json.loads(event) if isinstance(event, str) else event
        except (json.JSONDecodeError, TypeError):
            return

        if msg.get("event") != "posted":
            return

        data = msg.get("data", {})
        raw_post = data.get("post", "")
        try:
            post = json.loads(raw_post) if isinstance(raw_post, str) else raw_post
        except (json.JSONDecodeError, TypeError):
            return

        self._handle_post(post)

    def _handle_post(self, post: dict) -> None:
        """Process a new post: check trigger, call inference, reply."""
        # Ignore own messages
        if post.get("user_id") == self.bot_user_id:
            return

        channel_id = post.get("channel_id", "")
        message = post.get("message", "").strip()
        if not message:
            return

        is_dm = self._is_direct_message(channel_id)
        triggered = is_dm or self.trigger.lower() in message.lower()
        if not triggered:
            return

        # Strip the trigger from the message for cleaner input
        if not is_dm and self.trigger.lower() in message.lower():
            message = message.replace(self.trigger, "").strip()
            if not message:
                message = "Hello"

        # Build conversation history for this channel
        self.history[channel_id].append({"role": "user", "content": message})

        # Trim history
        if len(self.history[channel_id]) > self.max_history:
            self.history[channel_id] = self.history[channel_id][-self.max_history:]

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.history[channel_id])

        # Call inference
        try:
            reply = self._call_inference(messages)
        except Exception as e:
            reply = f"Inference error: {e}"

        # Track assistant reply in history
        self.history[channel_id].append({"role": "assistant", "content": reply})

        # Post reply as thread response
        root_id = post.get("root_id") or post.get("id", "")
        self.driver.posts.create_post({
            "channel_id": channel_id,
            "message": reply,
            "root_id": root_id,
        })

    def _call_inference(self, messages: list[dict[str, str]]) -> str:
        """Send chat completion request to local inference server."""
        url = f"{self.inference_url}/v1/chat/completions"
        payload = {
            "model": "local",
            "messages": messages,
            "stream": False,
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()

        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return "(no response from model)"

    def _is_direct_message(self, channel_id: str) -> bool:
        """Check if a channel is a direct message (type D or G)."""
        try:
            channel = self.driver.channels.get_channel(channel_id)
            return channel.get("type", "") in ("D", "G")
        except Exception:
            return False


def main() -> None:
    bridge = MattermostBridge()
    while True:
        try:
            bridge.start()
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Connection lost: {e}", file=sys.stderr)
            print("Reconnecting in 5 seconds...", file=sys.stderr)
            time.sleep(5)


if __name__ == "__main__":
    main()
