"""
07_notify.py — Email notifications via Resend SDK.

New in this version:
  • notify_training_progress() — sent every 2 hours by the background thread
    inside 03_train.py; shows current epoch, live metrics, time elapsed.
  • notify_error() — unchanged; called on crash by both shell and Python.
  • notify_folder_complete() — sent once per folder by 06_train_incremental.sh.

Setup:
    1. Sign up at resend.com and get an API key.
    2. Set in your .env file:
         RESEND_API_KEY=re_xxxx
         NOTIFY_TO=your@email.com
    3. Test: python src/07_notify.py --test
"""

import os
import sys
import json
import argparse
import base64
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))


# ── Config ────────────────────────────────────────────────────────────────────

def _cfg():
    return {
        "api_key": os.environ.get("RESEND_API_KEY", ""),
        "to":      os.environ.get("NOTIFY_TO", ""),
        "from":    "GeoSeg <onboarding@resend.dev>",
    }


# ── Core send ─────────────────────────────────────────────────────────────────

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
            params["attachments"] = [{"filename": "training_curves.png", "content": img_b64}]

        email = resend.Emails.send(params)
        print(f"[NOTIFY] ✓ Sent to {cfg['to']} (id={email.get('id','?')})", flush=True)
        return True
    except Exception as e:
        print(f"[NOTIFY] Failed: {e}", flush=True)
        return False


# ── Shared HTML helpers ───────────────────────────────────────────────────────

def _progress_bar(step, total):
    pct = int(step / total * 100)
    return f"[{'█' * step}{'░' * (total - step)}] {step}/{total} ({pct}%)"


def _miou_color(val_miou):
    if val_miou > 0.6:   return "#28a745"
    if val_miou > 0.4:   return "#fd7e14"
    return "#dc3545"


def _header(title, subtitle="", color="#1a1a2e"):
    return f"""
    <div style="background:{color};color:white;padding:20px;border-radius:8px 8px 0 0;">
        <h2 style="margin:0">{title}</h2>
        <p style="margin:5px 0;opacity:0.7">{subtitle}</p>
    </div>"""


def _footer():
    return """
    <div style="background:#6c757d;color:white;padding:10px;
                border-radius:0 0 8px 8px;font-size:12px;text-align:center;">
        GeoSeg Pipeline — Automated Notification
    </div>"""


# ── 1. Periodic progress email (every 2 hours) ────────────────────────────────

def notify_training_progress(folder_name, step, total, epoch, total_epochs,
                              train_loss, val_loss, train_miou, val_miou,
                              best_miou, checkpoint_path):
    """
    Called by the background thread in 03_train.py every NOTIFY_INTERVAL_HOURS.
    Shows a snapshot of the current training state.
    """
    cfg       = _cfg()
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    epoch_pct = int(epoch / total_epochs * 100)
    mc        = _miou_color(val_miou)

    curves = str(Path(checkpoint_path).parent / "training_curves.png")

    body = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:620px;margin:0 auto;">
    {_header("⏱️ GeoSeg — Training Progress Update", timestamp)}
    <div style="background:#f8f9fa;padding:20px;border:1px solid #dee2e6;">

      <h3 style="margin:0 0 10px">
        Folder: <b>{folder_name}</b> &nbsp;|&nbsp;
        Pipeline: {step}/{total}
      </h3>

      <div style="background:#e9ecef;padding:10px;border-radius:4px;font-family:monospace;margin-bottom:12px;">
        Pipeline: {_progress_bar(step, total)}<br>
        Epochs:   [{'█' * epoch_pct:░<100}] {epoch}/{total_epochs} ({epoch_pct}%)
      </div>

      <table style="width:100%;border-collapse:collapse;">
        <tr style="background:#dee2e6;">
          <th style="padding:8px;text-align:left">Metric</th>
          <th style="padding:8px;text-align:left">Current</th>
          <th style="padding:8px;text-align:left">Best</th>
        </tr>
        <tr>
          <td style="padding:8px"><b>Train Loss</b></td>
          <td>{train_loss:.4f}</td><td>—</td>
        </tr>
        <tr style="background:#f0f0f0">
          <td style="padding:8px"><b>Val Loss</b></td>
          <td>{val_loss:.4f}</td><td>—</td>
        </tr>
        <tr>
          <td style="padding:8px"><b>Train mIoU</b></td>
          <td>{train_miou:.4f} ({train_miou*100:.1f}%)</td><td>—</td>
        </tr>
        <tr style="background:#f0f0f0">
          <td style="padding:8px"><b>Val mIoU</b></td>
          <td style="color:{mc}"><b>{val_miou:.4f} ({val_miou*100:.1f}%)</b></td>
          <td style="color:{_miou_color(best_miou)}"><b>{best_miou:.4f}</b></td>
        </tr>
      </table>

      <p style="margin-top:12px;font-size:13px;color:#555">
        Training curves attached.
        Checkpoint: <code>{checkpoint_path}</code>
      </p>
    </div>
    {_footer()}
    </body></html>
    """
    send_email(
        cfg,
        f"[GeoSeg] ⏱️ Progress — {folder_name} epoch {epoch}/{total_epochs} "
        f"val_mIoU={val_miou:.3f}",
        body,
        attachment_path=curves,
    )


# ── 2. Folder-complete email ───────────────────────────────────────────────────

def notify_folder_complete(folder_name, step, total, train_loss, val_loss,
                            train_miou, val_miou, epochs, checkpoint_path):
    cfg       = _cfg()
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    mc        = _miou_color(val_miou)
    status    = "✅ All Done!" if step == total else "🔄 In Progress"

    history_html = ""
    history_path = Path(checkpoint_path).parent / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            h = json.load(f)
        best_ep = h["val_miou"].index(max(h["val_miou"])) + 1
        history_html = f"""
        <tr><td style="padding:8px"><b>Best Epoch</b></td><td>{best_ep}</td></tr>
        <tr><td style="padding:8px"><b>Total Epochs</b></td><td>{len(h['val_miou'])}</td></tr>
        """

    next_html = (
        f'<div style="margin-top:10px;padding:10px;background:#fff3cd;'
        f'border-radius:4px;color:#856404;">⏳ Next: CG_{step+1} queued...</div>'
        if step < total else
        '<div style="margin-top:10px;padding:10px;background:#d4edda;'
        'border-radius:4px;color:#155724;"><b>🎉 All folders complete! '
        'Final model ready.</b></div>'
    )

    body = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:620px;margin:0 auto;">
    {_header("🛰️ GeoSeg — Folder Complete", timestamp)}
    <div style="background:#f8f9fa;padding:20px;border:1px solid #dee2e6;">
      <h3 style="color:#495057">{folder_name} — {status}</h3>
      <div style="background:#e9ecef;padding:10px;border-radius:4px;font-family:monospace;">
        {_progress_bar(step, total)}
      </div>
      <table style="width:100%;border-collapse:collapse;margin-top:15px;">
        <tr style="background:#dee2e6;">
          <th style="padding:8px;text-align:left">Metric</th>
          <th style="padding:8px;text-align:left">Value</th>
        </tr>
        <tr><td style="padding:8px"><b>Train Loss</b></td><td>{train_loss:.4f}</td></tr>
        <tr style="background:#f0f0f0"><td style="padding:8px"><b>Val Loss</b></td><td>{val_loss:.4f}</td></tr>
        <tr><td style="padding:8px"><b>Train mIoU</b></td><td>{train_miou:.4f} ({train_miou*100:.1f}%)</td></tr>
        <tr style="background:#f0f0f0">
          <td style="padding:8px"><b>Val mIoU</b></td>
          <td style="color:{mc}"><b>{val_miou:.4f} ({val_miou*100:.1f}%)</b></td>
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
    {_footer()}
    </body></html>
    """
    curves = str(Path(checkpoint_path).parent / "training_curves.png")
    send_email(
        cfg,
        f"[GeoSeg] ✅ {folder_name} done ({step}/{total}) — val_mIoU={val_miou:.3f}",
        body,
        attachment_path=curves,
    )


# ── 3. Error / crash email ─────────────────────────────────────────────────────

def notify_error(folder_name, step, total, error_msg):
    cfg       = _cfg()
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    body = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:620px;">
    {_header("❌ GeoSeg Training Error", timestamp, color="#dc3545")}
    <div style="padding:20px;border:1px solid #f5c6cb;">
      <h3>Failed at: <b>{folder_name}</b> (Step {step}/{total})</h3>
      <pre style="background:#f8d7da;padding:12px;border-radius:4px;
                  overflow-x:auto;font-size:12px;">{error_msg}</pre>
      <p>The checkpoint at <b>checkpoints/best_model.pt</b> is safe.</p>
      <p>
        To resume from the last good folder:<br>
        <code>docker compose run --rm train-all --from {step}</code>
      </p>
    </div>
    {_footer()}
    </body></html>
    """
    send_email(cfg, f"[GeoSeg] ❌ ERROR at {folder_name} ({step}/{total})", body)


# ── 4. Test email ─────────────────────────────────────────────────────────────

def test_email():
    cfg = _cfg()
    print(f"  API Key : {cfg['api_key'][:8]}..." if cfg["api_key"] else "  API Key : NOT SET")
    print(f"  To      : {cfg['to']}")
    if not cfg["api_key"]:
        print("[ERROR] RESEND_API_KEY not set")
        sys.exit(1)
    body = """
    <html><body style="font-family:Arial,sans-serif;max-width:500px;">
    <div style="background:#28a745;color:white;padding:20px;border-radius:8px;">
        <h2>✅ GeoSeg Email Test</h2>
        <p>Notifications are working correctly.</p>
    </div>
    </body></html>
    """
    success = send_email(cfg, "[GeoSeg] ✅ Test Email", body)
    sys.exit(0 if success else 1)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",        action="store_true")
    parser.add_argument("--error",       action="store_true")
    parser.add_argument("--folder",      type=str,   default="")
    parser.add_argument("--step",        type=int,   default=0)
    parser.add_argument("--total",       type=int,   default=4)
    parser.add_argument("--train-loss",  type=float, default=0.0)
    parser.add_argument("--val-loss",    type=float, default=0.0)
    parser.add_argument("--train-miou",  type=float, default=0.0)
    parser.add_argument("--val-miou",    type=float, default=0.0)
    parser.add_argument("--epochs",      type=int,   default=0)
    parser.add_argument("--checkpoint",  type=str,   default="checkpoints/best_model.pt")
    parser.add_argument("--error-msg",   type=str,   default="")
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
            args.epochs, args.checkpoint,
        )