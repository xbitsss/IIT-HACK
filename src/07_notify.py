"""
07_notify.py — Email notifications via Resend SDK.

Functions
─────────
  notify_run_start()          — sent at the very start of training
  notify_training_progress()  — sent every 2 hours by background thread
  notify_milestone()          — sent at 25/50/75% of epochs + early stopping
  notify_folder_complete()    — sent once per shard / run
  notify_error()              — called on crash by shell and Python

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
from datetime import datetime

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
        print(f"[NOTIFY] Sent to {cfg['to']} (id={email.get('id','?')})", flush=True)
        return True
    except Exception as e:
        print(f"[NOTIFY] Failed: {e}", flush=True)
        return False


# ── Shared HTML helpers ───────────────────────────────────────────────────────

def _progress_bar(step, total):
    pct    = int(step / total * 100)
    filled = "█" * step
    empty  = "░" * (total - step)
    return f"[{filled}{empty}] {step}/{total} ({pct}%)"


def _miou_color(val_miou):
    if val_miou > 0.6:   return "#28a745"
    if val_miou > 0.4:   return "#fd7e14"
    return "#dc3545"


def _header(title, subtitle="", color="#1a1a2e"):
    return (
        f'<div style="background:{color};color:white;padding:20px;border-radius:8px 8px 0 0;">'
        f'<h2 style="margin:0">{title}</h2>'
        f'<p style="margin:5px 0;opacity:0.7">{subtitle}</p>'
        f'</div>'
    )


def _footer():
    return (
        '<div style="background:#6c757d;color:white;padding:10px;'
        'border-radius:0 0 8px 8px;font-size:12px;text-align:center;">'
        'GeoSeg Pipeline — Automated Notification'
        '</div>'
    )


def _version_badge(version, message=""):
    if not version:
        return ""
    msg_part = ""
    if message:
        msg_part = f"<br><small style='opacity:0.8'>{message}</small>"
    return (
        f'<div style="display:inline-block;background:#0d6efd;color:white;'
        f'padding:4px 12px;border-radius:12px;font-size:13px;margin-bottom:10px;">'
        f'v{version}{msg_part}'
        f'</div>'
    )


# ── 1. Run start notification ──────────────────────────────────────────────────

def notify_run_start(model_type, version, run_message, mode, folder_name,
                     step, total, config_summary=""):
    cfg        = _cfg()
    timestamp  = datetime.now().strftime("%d %b %Y, %I:%M %p")
    mode_color = {"FRESH": "#28a745", "RESUME": "#0d6efd", "INIT-WEIGHTS": "#fd7e14"}.get(mode, "#6c757d")
    cfg_row    = ""
    if config_summary:
        cfg_row = (
            f"<tr><td style='padding:8px'><b>Config</b></td>"
            f"<td><small>{config_summary}</small></td></tr>"
        )

    body = (
        f"<html><body style='font-family:Arial,sans-serif;max-width:620px;margin:0 auto;'>"
        f"{_header('🚀 GeoSeg — Training Started', timestamp)}"
        f"<div style='background:#f8f9fa;padding:20px;border:1px solid #dee2e6;'>"
        f"{_version_badge(version, run_message)}"
        f"<table style='width:100%;border-collapse:collapse;margin-top:10px;'>"
        f"<tr style='background:#dee2e6;'><th style='padding:8px;text-align:left'>Field</th>"
        f"<th style='padding:8px;text-align:left'>Value</th></tr>"
        f"<tr><td style='padding:8px'><b>Model type</b></td><td>{model_type}</td></tr>"
        f"<tr style='background:#f0f0f0'><td style='padding:8px'><b>Version</b></td>"
        f"<td><b>v{version}</b></td></tr>"
        f"<tr><td style='padding:8px'><b>Mode</b></td>"
        f"<td><span style='color:{mode_color}'><b>{mode}</b></span></td></tr>"
        f"<tr style='background:#f0f0f0'><td style='padding:8px'><b>Shard</b></td>"
        f"<td>{folder_name} ({step}/{total})</td></tr>"
        f"{cfg_row}"
        f"</table>"
        f"<p style='margin-top:12px;color:#555;font-size:13px;'>"
        f"Progress emails every 2 hours. Milestone emails at 25%, 50%, 75%, 100%.</p>"
        f"</div>"
        f"{_footer()}"
        f"</body></html>"
    )
    send_email(
        cfg,
        f"[GeoSeg] 🚀 {model_type} v{version} started — {folder_name} ({mode})",
        body,
    )


# ── 2. Milestone notification ──────────────────────────────────────────────────

def notify_milestone(stage, milestone_name, version=None, run_message="",
                     val_miou=None, best_miou=None, epoch=None, total_epochs=None,
                     details="", checkpoint_path="", attachment_path=None):
    cfg       = _cfg()
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

    emoji_map = {
        "preprocess_done":  "📦",
        "replay_saved":     "💾",
        "25pct_done":       "📊",
        "50pct_done":       "📊",
        "75pct_done":       "📊",
        "training_complete":"✅",
        "early_stopping":   "🛑",
        "shard_done":       "✅",
        "all_done":         "🎉",
        "specialist_done":  "🎯",
    }
    emoji = emoji_map.get(milestone_name, "🔔")

    miou_row = ""
    if val_miou is not None:
        mc = _miou_color(val_miou)
        miou_row = (
            f"<tr style='background:#f0f0f0'>"
            f"<td style='padding:8px'><b>Val mIoU</b></td>"
            f"<td style='color:{mc}'><b>{val_miou:.4f} ({val_miou*100:.1f}%)</b></td></tr>"
        )
        if best_miou is not None:
            bc = _miou_color(best_miou)
            miou_row += (
                f"<tr><td style='padding:8px'><b>Best so far</b></td>"
                f"<td style='color:{bc}'><b>{best_miou:.4f}</b></td></tr>"
            )

    epoch_row = ""
    if epoch is not None and total_epochs is not None:
        pct = int(epoch / total_epochs * 100)
        epoch_row = (
            f"<tr><td style='padding:8px'><b>Epoch</b></td>"
            f"<td>{epoch}/{total_epochs} ({pct}%)</td></tr>"
        )

    ckpt_row = ""
    if checkpoint_path:
        ckpt_row = (
            f"<tr style='background:#f0f0f0'>"
            f"<td style='padding:8px'><b>Checkpoint</b></td>"
            f"<td><code style='font-size:12px'>{checkpoint_path}</code></td></tr>"
        )

    details_row = ""
    if details:
        details_row = (
            f"<tr><td colspan='2' style='padding:8px;color:#555;font-size:13px;'>"
            f"{details}</td></tr>"
        )

    ver_str = str(version) if version else "?"
    milestone_label = milestone_name.replace("_", " ").title()

    body = (
        f"<html><body style='font-family:Arial,sans-serif;max-width:620px;margin:0 auto;'>"
        f"{_header(emoji + ' GeoSeg — ' + stage.title() + ' Milestone', timestamp)}"
        f"<div style='background:#f8f9fa;padding:20px;border:1px solid #dee2e6;'>"
        f"{_version_badge(version, run_message) if version else ''}"
        f"<h3 style='margin:0 0 12px;color:#333'>{milestone_label}</h3>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<tr style='background:#dee2e6;'>"
        f"<th style='padding:8px;text-align:left'>Field</th>"
        f"<th style='padding:8px;text-align:left'>Value</th></tr>"
        f"<tr><td style='padding:8px'><b>Stage</b></td><td>{stage}</td></tr>"
        f"{epoch_row}{miou_row}{ckpt_row}{details_row}"
        f"</table></div>"
        f"{_footer()}"
        f"</body></html>"
    )
    send_email(
        cfg,
        f"[GeoSeg] {emoji} {stage} v{ver_str} — {milestone_label}",
        body,
        attachment_path=attachment_path,
    )


# ── 3. Periodic progress email (every 2 hours) ────────────────────────────────

def notify_training_progress(folder_name, step, total, epoch, total_epochs,
                              train_loss, val_loss, train_miou, val_miou,
                              best_miou, checkpoint_path,
                              version=None, run_message=""):
    cfg       = _cfg()
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    epoch_pct = int(epoch / total_epochs * 100)
    mc        = _miou_color(val_miou)
    bmc       = _miou_color(best_miou)

    # Build epoch bar without backslash in f-string (Python < 3.12 safe)
    bar_filled = "█" * (epoch_pct // 2)
    bar_empty  = "░" * (50 - epoch_pct // 2)
    epoch_bar  = f"[{bar_filled}{bar_empty}] {epoch}/{total_epochs} ({epoch_pct}%)"
    pipe_bar   = _progress_bar(step, total)

    curves = str(Path(checkpoint_path).parent / "training_curves.png")

    body = (
        f"<html><body style='font-family:Arial,sans-serif;max-width:620px;margin:0 auto;'>"
        f"{_header('⏱️ GeoSeg — Training Progress Update', timestamp)}"
        f"<div style='background:#f8f9fa;padding:20px;border:1px solid #dee2e6;'>"
        f"{_version_badge(version, run_message) if version else ''}"
        f"<h3 style='margin:0 0 10px'>Folder: <b>{folder_name}</b> | Pipeline: {step}/{total}</h3>"
        f"<div style='background:#e9ecef;padding:10px;border-radius:4px;"
        f"font-family:monospace;margin-bottom:12px;font-size:13px;'>"
        f"Pipeline: {pipe_bar}<br>Epochs:   {epoch_bar}</div>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<tr style='background:#dee2e6;'>"
        f"<th style='padding:8px;text-align:left'>Metric</th>"
        f"<th style='padding:8px;text-align:left'>Current</th>"
        f"<th style='padding:8px;text-align:left'>Best</th></tr>"
        f"<tr><td style='padding:8px'><b>Train Loss</b></td>"
        f"<td>{train_loss:.4f}</td><td>—</td></tr>"
        f"<tr style='background:#f0f0f0'>"
        f"<td style='padding:8px'><b>Val Loss</b></td><td>{val_loss:.4f}</td><td>—</td></tr>"
        f"<tr><td style='padding:8px'><b>Train mIoU</b></td>"
        f"<td>{train_miou:.4f} ({train_miou*100:.1f}%)</td><td>—</td></tr>"
        f"<tr style='background:#f0f0f0'>"
        f"<td style='padding:8px'><b>Val mIoU</b></td>"
        f"<td style='color:{mc}'><b>{val_miou:.4f} ({val_miou*100:.1f}%)</b></td>"
        f"<td style='color:{bmc}'><b>{best_miou:.4f}</b></td></tr>"
        f"</table>"
        f"<p style='margin-top:12px;font-size:13px;color:#555'>"
        f"Training curves attached. Checkpoint: <code>{checkpoint_path}</code></p>"
        f"</div>"
        f"{_footer()}"
        f"</body></html>"
    )
    send_email(
        cfg,
        f"[GeoSeg] ⏱️ Progress — {folder_name} epoch {epoch}/{total_epochs} "
        f"val_mIoU={val_miou:.3f}",
        body,
        attachment_path=curves,
    )


# ── 4. Shard-complete email ───────────────────────────────────────────────────

def notify_folder_complete(folder_name, step, total, train_loss, val_loss,
                            train_miou, val_miou, epochs, checkpoint_path,
                            version=None, run_message="", per_class_iou=None):
    cfg       = _cfg()
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    mc        = _miou_color(val_miou)
    status    = "✅ All Done!" if step == total else "🔄 In Progress"

    history_html = ""
    history_path = Path(checkpoint_path).parent / "history.json"
    if history_path.exists():
        try:
            with open(history_path) as f:
                h = json.load(f)
            best_ep = h["val_miou"].index(max(h["val_miou"])) + 1
            history_html = (
                f"<tr><td style='padding:8px'><b>Best Epoch</b></td><td>{best_ep}</td></tr>"
                f"<tr><td style='padding:8px'><b>Total Epochs</b></td><td>{len(h['val_miou'])}</td></tr>"
            )
        except Exception:
            pass

    # Per-class IoU table — build without backslash-in-f-string (Python < 3.12 safe)
    per_class_html = ""
    if per_class_iou:
        # Pre-compute row style to avoid backslash in f-string expression
        even_style = ""
        odd_style  = "style='background:#f0f0f0'"
        rows = ""
        for i, (label, iou) in enumerate(per_class_iou.items()):
            row_style = odd_style if i % 2 else even_style
            rows += (
                f"<tr {row_style}>"
                f"<td style='padding:6px'>{label}</td>"
                f"<td style='padding:6px'>{float(iou):.4f}</td></tr>"
            )
        per_class_html = (
            f"<h4 style='margin:16px 0 6px'>Per-class IoU</h4>"
            f"<table style='width:100%;border-collapse:collapse;'>{rows}</table>"
        )

    next_html = (
        f"<div style='margin-top:10px;padding:10px;background:#fff3cd;"
        f"border-radius:4px;color:#856404;'>⏳ Next shard queued...</div>"
        if step < total else
        "<div style='margin-top:10px;padding:10px;background:#d4edda;"
        "border-radius:4px;color:#155724;'><b>🎉 All shards complete!</b></div>"
    )

    body = (
        f"<html><body style='font-family:Arial,sans-serif;max-width:620px;margin:0 auto;'>"
        f"{_header('🛰️ GeoSeg — Shard Complete', timestamp)}"
        f"<div style='background:#f8f9fa;padding:20px;border:1px solid #dee2e6;'>"
        f"{_version_badge(version, run_message) if version else ''}"
        f"<h3 style='color:#495057'>{folder_name} — {status}</h3>"
        f"<div style='background:#e9ecef;padding:10px;border-radius:4px;"
        f"font-family:monospace;font-size:13px;'>{_progress_bar(step, total)}</div>"
        f"<table style='width:100%;border-collapse:collapse;margin-top:15px;'>"
        f"<tr style='background:#dee2e6;'>"
        f"<th style='padding:8px;text-align:left'>Metric</th>"
        f"<th style='padding:8px;text-align:left'>Value</th></tr>"
        f"<tr><td style='padding:8px'><b>Train Loss</b></td><td>{train_loss:.4f}</td></tr>"
        f"<tr style='background:#f0f0f0'>"
        f"<td style='padding:8px'><b>Val Loss</b></td><td>{val_loss:.4f}</td></tr>"
        f"<tr><td style='padding:8px'><b>Train mIoU</b></td>"
        f"<td>{train_miou:.4f} ({train_miou*100:.1f}%)</td></tr>"
        f"<tr style='background:#f0f0f0'>"
        f"<td style='padding:8px'><b>Val mIoU</b></td>"
        f"<td style='color:{mc}'><b>{val_miou:.4f} ({val_miou*100:.1f}%)</b></td></tr>"
        f"<tr><td style='padding:8px'><b>Epochs</b></td><td>{epochs}</td></tr>"
        f"{history_html}"
        f"</table>"
        f"{per_class_html}"
        f"<p><i>Training curves attached (if available).</i></p>"
        f"<div style='margin-top:15px;padding:10px;background:#d4edda;"
        f"border-radius:4px;color:#155724;'><b>Checkpoint:</b> {checkpoint_path}</div>"
        f"{next_html}"
        f"</div>"
        f"{_footer()}"
        f"</body></html>"
    )
    curves = str(Path(checkpoint_path).parent / "training_curves.png")
    send_email(
        cfg,
        f"[GeoSeg] ✅ {folder_name} done ({step}/{total}) — val_mIoU={val_miou:.3f}",
        body,
        attachment_path=curves,
    )


# ── 5. Error / crash email ────────────────────────────────────────────────────

def notify_error(folder_name, step, total, error_msg, version=None, run_message=""):
    cfg       = _cfg()
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    resume_cmd = f"--from {step}" if step > 0 else "--from 1"

    body = (
        f"<html><body style='font-family:Arial,sans-serif;max-width:620px;'>"
        f"{_header('❌ GeoSeg Training Error', timestamp, color='#dc3545')}"
        f"<div style='padding:20px;border:1px solid #f5c6cb;'>"
        f"{_version_badge(version, run_message) if version else ''}"
        f"<h3>Failed at: <b>{folder_name}</b> (Step {step}/{total})</h3>"
        f"<pre style='background:#f8d7da;padding:12px;border-radius:4px;"
        f"overflow-x:auto;font-size:12px;'>{error_msg}</pre>"
        f"<p>The checkpoint at <b>checkpoints/best_model.pt</b> is safe.</p>"
        f"<p>To resume from the last good shard:<br>"
        f"<code>docker compose run --rm train-all {resume_cmd}</code><br>"
        f"Or for specialist only:<br>"
        f"<code>docker compose run --rm specialist</code></p>"
        f"<p>See <b>PIPELINE_GUIDE.md</b> for recovery steps.</p>"
        f"</div>"
        f"{_footer()}"
        f"</body></html>"
    )
    send_email(cfg, f"[GeoSeg] ❌ ERROR at {folder_name} ({step}/{total})", body)


# ── 6. Test email ─────────────────────────────────────────────────────────────

def test_email():
    cfg = _cfg()
    if cfg["api_key"]:
        print(f"  API Key : {cfg['api_key'][:8]}...")
    else:
        print("  API Key : NOT SET")
    print(f"  To      : {cfg['to']}")
    if not cfg["api_key"]:
        print("[ERROR] RESEND_API_KEY not set")
        sys.exit(1)
    body = (
        "<html><body style='font-family:Arial,sans-serif;max-width:500px;'>"
        "<div style='background:#28a745;color:white;padding:20px;border-radius:8px;'>"
        "<h2>✅ GeoSeg Email Test</h2>"
        "<p>Notifications are working correctly.</p>"
        "<p>You will receive emails for: run start, 2-hour progress updates, "
        "25%/50%/75%/100% milestones, shard complete, errors.</p>"
        "</div></body></html>"
    )
    success = send_email(cfg, "[GeoSeg] ✅ Test Email", body)
    sys.exit(0 if success else 1)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoSeg notification sender")
    parser.add_argument("--test",            action="store_true",
                        help="Send a test email to verify configuration")
    parser.add_argument("--error",           action="store_true",
                        help="Send a crash/error notification")
    parser.add_argument("--milestone",       action="store_true",
                        help="Send a training milestone notification")
    parser.add_argument("--folder",          type=str,   default="",
                        help="Folder/stage name")
    parser.add_argument("--step",            type=int,   default=0,
                        help="Current pipeline step")
    parser.add_argument("--total",           type=int,   default=2,
                        help="Total pipeline steps")
    parser.add_argument("--train-loss",      type=float, default=0.0)
    parser.add_argument("--val-loss",        type=float, default=0.0)
    parser.add_argument("--train-miou",      type=float, default=0.0)
    parser.add_argument("--val-miou",        type=float, default=0.0)
    parser.add_argument("--epochs",          type=int,   default=0)
    parser.add_argument("--checkpoint",      type=str,   default="checkpoints/best_model.pt")
    parser.add_argument("--error-msg",       type=str,   default="",
                        help="Error message body (used with --error)")
    parser.add_argument("--milestone-name",  type=str,   default="shard_done",
                        help="Milestone identifier e.g. 25pct_done, shard_done")
    parser.add_argument("--details",         type=str,   default="",
                        help="Extra details for milestone emails")
    # Version and run-message: always present so the shell can always pass them
    parser.add_argument("--version",         type=int,   default=None,
                        help="Run version number (optional)")
    parser.add_argument("--run-message",     type=str,   default="",
                        help="Human-readable description of this run")
    args = parser.parse_args()

    if args.test:
        test_email()
    elif args.error:
        notify_error(
            args.folder, args.step, args.total, args.error_msg,
            version=args.version, run_message=args.run_message,
        )
    elif args.milestone:
        notify_milestone(
            stage=args.folder or "pipeline",
            milestone_name=args.milestone_name,
            version=args.version,
            run_message=args.run_message,
            details=args.details,
        )
    else:
        notify_folder_complete(
            args.folder, args.step, args.total,
            args.train_loss, args.val_loss,
            args.train_miou, args.val_miou,
            args.epochs, args.checkpoint,
            version=args.version,
            run_message=args.run_message,
        )