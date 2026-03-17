"""Telegram trade notifications — credentials from env vars only."""
import logging
import urllib.request
import urllib.parse
import urllib.error
import json

log = logging.getLogger(__name__)


class Telegram:
    def __init__(self, config: dict) -> None:
        self.enabled = config.get("telegram_enabled", False)
        self.token = config.get("telegram_bot_token", "")
        self.chat_id = config.get("telegram_chat_id", "")

        # Disable if token/chat_id still contain placeholder
        if "${" in str(self.token) or "${" in str(self.chat_id):
            log.warning("Telegram credentials not set in env — notifications disabled")
            self.enabled = False

        if self.enabled and (not self.token or not self.chat_id):
            log.warning("Telegram token or chat_id missing — notifications disabled")
            self.enabled = False

    def send(self, text: str) -> bool:
        """Send a message. Returns True on success."""
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
        }).encode()
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                return result.get("ok", False)
        except urllib.error.URLError as e:
            log.warning("Telegram send failed: %s", e)
            return False
        except Exception as e:
            log.warning("Telegram unexpected error: %s", e)
            return False
