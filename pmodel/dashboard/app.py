"""Flask dashboard. Run locally with `python -m pmodel dashboard`, or export
every page to static HTML with `python -m pmodel build-site` (open site/index.html
in any browser — no server needed, works on a locked-down school computer)."""
import base64
from datetime import datetime, timezone

from flask import Flask, render_template

from pmodel import config
from pmodel.dashboard import charts, data
from pmodel.model.train import MODELS_DIR
from pmodel.trading import ledger

REPO_URL = "https://github.com/matthewcanzani-lab/polymarket-paper-trading"
PAGES = [("index.html", "Overview"), ("markets.html", "Live Markets"), ("edges.html", "Edge Finder"),
         ("portfolio.html", "Paper Portfolio"), ("stats.html", "Model Stats"),
         ("results/", "Results")]


def create_app(ledger_path=None, scores_path=None, models_dir=MODELS_DIR, weather_scores_path=None):
    app = Flask(__name__)

    def ctx(active, root=""):
        last = ledger.connect(ledger_path).execute(
            "SELECT MAX(finished_at) FROM runs").fetchone()[0]
        age = data.age_hours(last)
        return {"pages": PAGES, "active": active, "root": root, "repo_url": REPO_URL,
                "cfg": config, "age_hours": data.age_hours,
                "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "last_run_iso": last, "last_run_et": data.eastern(last),
                "stale_at_build": age is None or age > config.STALE_AFTER_HOURS}

    def conn():
        return ledger.connect(ledger_path)

    @app.route("/")
    @app.route("/index.html")
    def index():
        c = conn()
        fs = data.forward_stats(c)
        scores = data.load_scores(scores_path)
        return render_template("index.html", **ctx("index.html"), fs=fs, scores=scores,
                               criteria=data.success_criteria(fs), runs_by_strategy=data.latest_runs(c),
                               weather=data.load_weather_scores(weather_scores_path),
                               integrity=ledger.verify(c),
                               runs=c.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    @app.route("/markets.html")
    def markets():
        scores = data.load_scores(scores_path)
        return render_template("markets.html", **ctx("markets.html"), scores=scores,
                               rows=data.live_markets(scores))

    @app.route("/edges.html")
    def edges():
        scores = data.load_scores(scores_path)
        weather = data.load_weather_scores(weather_scores_path)
        return render_template("edges.html", **ctx("edges.html"), scores=scores,
                               rows=data.edges(scores), weather=weather,
                               weather_rows=data.edges(weather))

    @app.route("/portfolio.html")
    def portfolio():
        c = conn()
        all_bets = data.bets(c)
        return render_template(
            "portfolio.html", **ctx("portfolio.html"), fs=data.forward_stats(c),
            pnl_svg=charts.pnl_chart(data.pnl_curve(c), config.BANKROLL_START_USD),
            open_bets=[b for b in all_bets if b["settled_at"] is None],
            open_by_strategy=data.open_by_strategy(c),
            settled=[b for b in all_bets if b["settled_at"] is not None])

    @app.route("/stats.html")
    def stats():
        c = conn()
        fs = data.forward_stats(c)
        versions = data.backtests(models_dir)
        current = next((v for v in versions if not v["superseded"]), None)
        bt_img = None
        if current and (current["path"] / "calibration.png").exists():
            bt_img = base64.b64encode((current["path"] / "calibration.png").read_bytes()).decode()
        return render_template("stats.html", **ctx("stats.html"), fs=fs,
                               fwd_cal_svg=charts.calibration_chart(fs["calibration"]),
                               criteria=data.success_criteria(fs), current=current,
                               versions=versions, bt_img=bt_img)

    @app.route("/results/")
    def results():
        c = conn()
        fs = data.forward_stats(c)
        return render_template("results.html", **ctx("results/", root="../"),
                               r=data.results_summary(c),
                               cal_svg=charts.calibration_chart(fs["calibration"], compact=True))

    return app


def build_site(out_dir, **kw):
    """Render every page to static HTML files in out_dir."""
    app = create_app(**kw)
    out_dir.mkdir(parents=True, exist_ok=True)
    with app.test_client() as client:
        for page, _ in PAGES:
            resp = client.get(f"/{page}")
            assert resp.status_code == 200, (page, resp.status_code)
            # "results/" becomes results/index.html, so it is served at /results/
            target = out_dir / page / "index.html" if page.endswith("/") else out_dir / page
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(resp.data)
    return out_dir
