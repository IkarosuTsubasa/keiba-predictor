import argparse
import json
from datetime import datetime

from ntfy_notifier import fcm_topic


def parse_args():
    parser = argparse.ArgumentParser(description="Send a test FCM notification through firebase-admin.")
    parser.add_argument("--topic", default="", help="Override FCM topic.")
    parser.add_argument("--title", default="🐴予測が完了しました", help="Notification title.")
    parser.add_argument("--body", default="", help="Notification body.")
    return parser.parse_args()


def main():
    args = parse_args()
    topic = str(args.topic or fcm_topic() or "").strip()
    if not topic:
        raise SystemExit("FCM topic is empty. Set PIPELINE_FCM_TOPIC or pass --topic.")

    body = str(args.body or f"#{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} テスト通知です").strip()

    from firebase_admin import messaging

    from ntfy_notifier import _get_firebase_app

    app = _get_firebase_app()
    message = messaging.Message(
        topic=topic,
        notification=messaging.Notification(
            title=str(args.title or "").strip(),
            body=body,
        ),
        android=messaging.AndroidConfig(priority="high"),
    )
    message_id = str(messaging.send(message, app=app) or "").strip()
    print(
        json.dumps(
            {
                "ok": True,
                "topic": topic,
                "title": str(args.title or "").strip(),
                "body": body,
                "message_id": message_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
