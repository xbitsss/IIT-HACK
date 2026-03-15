"""
07_notify.py — Email notifications via Resend SDK.
Setup: sign up at resend.com, get API key, set in .env:
    RESEND_API_KEY=re_xxxx
    NOTIFY_TO=your@email.com
Test: python src/07_notify.py --test
"""

import os
import sys
import json
import argparse
import base64
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))


def load_cfg():
    return {
        "api_key": os.environ.get("RESEND_API_KEY", ""),
        "to":      os.environ.get("NOTIFY_TO", ""),
        "from":    "GeoSeg <onboarding@resend.dev>",
    }


def send_email(cfg, subject, body_html, attachment_path=None):
    if not cfg["api_key"] or not cfg["to"]:
        print("[NOTIFY] Not configured — set RESEND_API_KEY and NOTIFY_TO in .env", flush=True)
        return False

    try:
        import resend
        resend.api_key = cfg["api_key"]

        params = {
            "from":    cfg["from"],
            "to":      [cfg["to"]],
            "subject": subject,
            "html":    body_html,
        }

        if attachment_path and Path(attachment_path).exists():
            with open(attachment_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            params["attachments"] = [{
                "filename": "training_curves.png",
                "content":  img_b64,
            }]

        email = resend.Emails.send(params)
        print(f"[NOTIFY] ✓ Email sent to {cfg['to']} (id={email.get('id','?')})", flush=True)
        return True

    except Exception as e:
        print(f"[NOTIFY] Failed: {e}", flush=True)
        return False


def notify_folder_complete(folder_name, step, total, train_loss, val_loss,
                            train_miou, val_miou, epochs, checkpoint_path):
    cfg          = load_cfg()
    progress_bar = "█" * step + "░" * (total - step)
    pct          = int(step / total * 100)
    status       = "✅ Complete" if step == total else "🔄 In Progress"
    timestamp    = datetime.now().strftime("%d %b %Y, %I:%M %p")
    miou_color   = "#28a745" if val_miou > 0.6 else "#fd7e14" if val_miou > 0.4 else "#dc3545"

    history_html = ""
    history_path = Path(checkpoint_path).parent / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        best_epoch   = history["val_miou"].index(max(history["val_miou"])) + 1
        history_html = f"""
        <tr><td><b>Best Epoch</b></td><td>{best_epoch}</td></tr>
        <tr><td><b>Total Epochs Run</b></td><td>{len(history['val_miou'])}</td></tr>
        """

    next_html = f'<div style="margin-top:10px;padding:10px;background:#fff3cd;border-radius:4px;color:#856404;">⏳ Next: CG_{step+1} starting...</div>' \
                if step < total else \
                '<div style="margin-top:10px;padding:10px;background:#d4edda;border-radius:4px;color:#155724;"><b>🎉 All folders complete! Final model ready.</b></div>'

    body = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
    <div style="background:#1a1a2e;color:white;padding:20px;border-radius:8px 8px 0 0;">
        <h2 style="margin:0">🛰️ GeoSeg Training Update</h2>
        <p style="margin:5px 0;opacity:0.7">{timestamp}</p>
    </div>
    <div style="background:#f8f9fa;padding:20px;border:1px solid #dee2e6;">
        <h3 style="color:#495057">Folder: {folder_name} — {status}</h3>
        <div style="background:#e9ecef;padding:10px;border-radius:4px;font-family:monospace;">
            [{progress_bar}] {step}/{total} ({pct}%)
        </div>
        <table style="width:100%;border-collapse:collapse;margin-top:15px;">
            <tr style="background:#dee2e6;">
                <th style="padding:8px;text-align:left">Metric</th>
                <th style="padding:8px;text-align:left">Value</th>
            </tr>
            <tr><td style="padding:8px"><b>Train Loss</b></td><td>{train_loss:.4f}</td></tr>
            <tr style="background:#f8f9fa"><td style="padding:8px"><b>Val Loss</b></td><td>{val_loss:.4f}</td></tr>
            <tr><td style="padding:8px"><b>Train mIoU</b></td><td>{train_miou:.4f} ({train_miou*100:.1f}%)</td></tr>
            <tr style="background:#f8f9fa">
                <td style="padding:8px"><b>Val mIoU</b></td>
                <td style="color:{miou_color}"><b>{val_miou:.4f} ({val_miou*100:.1f}%)</b></td>
            </tr>
            <tr><td style="padding:8px"><b>Epochs</b></td><td>{epochs}</td></tr>
            {history_html}
        </table>
        <p><i>Training curves attached.</i></p>
        <div style="margin-top:15px;padding:10px;background:#d4edda;border-radius:4px;color:#155724;">
            <b>Checkpoint:</b> {checkpoint_path}
        </div>
        {next_html}
    </div>
    <div style="background:#6c757d;color:white;padding:10px;border-radius:0 0 8px 8px;font-size:12px;">
        GeoSeg Pipeline — Automated Notification
    </div>
    </body></html>
    """

    curves = str(Path(checkpoint_path).parent / "training_curves.png")
    send_email(cfg, f"[GeoSeg] {folder_name} done ({step}/{total}) — val_mIoU={val_miou:.3f}",
               body, attachment_path=curves)


def notify_error(folder_name, step, total, error_msg):
    cfg       = load_cfg()
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    body = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:600px;">
    <div style="background:#dc3545;color:white;padding:20px;border-radius:8px 8px 0 0;">
        <h2>❌ GeoSeg Training Error</h2>
        <p style="opacity:0.7">{timestamp}</p>
    </div>
    <div style="padding:20px;border:1px solid #f5c6cb;">
        <h3>Failed at: {folder_name} (Step {step}/{total})</h3>
        <pre style="background:#f8d7da;padding:10px;border-radius:4px;">{error_msg}</pre>
        <p>Checkpoint is safe at <b>checkpoints/best_model.pt</b></p>
        <p>Resume: <code>docker compose run --rm train --resume</code></p>
    </div>
    </body></html>
    """
    send_email(cfg, f"[GeoSeg] ❌ ERROR at {folder_name} ({step}/{total})", body)


def test_email():
    cfg = load_cfg()
    print(f"  API Key: {cfg['api_key'][:8]}..." if cfg["api_key"] else "  API Key: NOT SET", flush=True)
    print(f"  To: {cfg['to']}", flush=True)
    if not cfg["api_key"]:
        print("[ERROR] RESEND_API_KEY not set", flush=True)
        sys.exit(1)
    body = """
    <html><body style="font-family:Arial,sans-serif;max-width:600px;">
    <div style="background:#28a745;color:white;padding:20px;border-radius:8px;">
        <h2>✅ GeoSeg Email Test</h2>
        <p>Email notifications are working!</p>
    </div>
    </body></html>
    """
    success = send_email(cfg, "[GeoSeg] ✅ Test Email", body)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",       action="store_true")
    parser.add_argument("--error",      action="store_true")
    parser.add_argument("--folder",     type=str,   default="")
    parser.add_argument("--step",       type=int,   default=0)
    parser.add_argument("--total",      type=int,   default=4)
    parser.add_argument("--train-loss", type=float, default=0.0)
    parser.add_argument("--val-loss",   type=float, default=0.0)
    parser.add_argument("--train-miou", type=float, default=0.0)
    parser.add_argument("--val-miou",   type=float, default=0.0)
    parser.add_argument("--epochs",     type=int,   default=0)
    parser.add_argument("--checkpoint", type=str,   default="checkpoints/best_model.pt")
    parser.add_argument("--error-msg",  type=str,   default="")
    args = parser.parse_args()

    if args.test:
        test_email()
    elif args.error:
        notify_error(args.folder, args.step, args.total, args.error_msg)
    else:
        notify_folder_complete(
            args.folder, args.step, args.total,
            args.train_loss, args.val_loss,
            args.train_miou, args.val_miou,
            args.epochs, args.checkpoint
        )