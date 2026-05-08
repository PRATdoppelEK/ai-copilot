"""
Notification utility — push alerts via ntfy.sh.
Free, open-source push notifications to any device. No account required.
"""

import logging
import requests

logger = logging.getLogger(__name__)


def send_notification(
    message: str,
    title:   str = "AI Copilot",
    topic:   str = "",
    tags:    str = "robot",
    priority: str = "default",
) -> bool:
    """
    Send a push notification via ntfy.sh.

    Setup:
        1. Install the ntfy app on your phone: https://ntfy.sh
        2. Subscribe to your unique topic (any string you choose)
        3. Set your topic in configs/config.yaml → notifications.ntfy_topic

    Args:
        message  : notification body text
        title    : notification title shown on device
        topic    : your ntfy topic (from config)
        tags     : emoji tags — see https://ntfy.sh/docs/emojis/
        priority : min | low | default | high | urgent

    Returns:
        True if notification sent successfully, False otherwise.
    """
    if not topic:
        logger.warning("ntfy topic not set — skipping notification")
        return False

    try:
        resp = requests.post(
            f"https://ntfy.sh/{topic}",
            data    = message.encode("utf-8"),
            headers = {
                "Title":    title,
                "Priority": priority,
                "Tags":     tags,
            },
            timeout = 8,
        )
        if resp.status_code == 200:
            logger.info("📱 Notification sent successfully")
            return True
        else:
            logger.warning(f"ntfy returned status {resp.status_code}")
            return False
    except requests.exceptions.Timeout:
        logger.warning("ntfy notification timed out")
        return False
    except Exception as e:
        logger.warning(f"ntfy notification failed: {e}")
        return False
