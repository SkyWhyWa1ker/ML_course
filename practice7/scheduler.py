"""
scheduler.py
------------
Runs batch_predict.run() automatically on a fixed interval
using APScheduler (BackgroundScheduler).

Default: every 5 minutes.
Change INTERVAL_MINUTES to adjust the schedule.

Usage:
    python scheduler.py
"""

import logging
import time

from apscheduler.schedulers.background import BackgroundScheduler

import batch_predict

INTERVAL_MINUTES = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scheduler")


def job() -> None:
    """Wrapper called by the scheduler on each tick."""
    batch_predict.run()


def main() -> None:
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        func=job,
        trigger="interval",
        minutes=INTERVAL_MINUTES,
        id="batch_prediction_job",
        name="BatchPrediction",
        replace_existing=True,
    )

    scheduler.start()
    log.info("Scheduler started — running every %d minute(s).", INTERVAL_MINUTES)

    for j in scheduler.get_jobs():
        log.info("  Job: %s | Next run: %s", j.name, j.next_run_time)

    log.info("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        log.info("Shutting down scheduler...")
        scheduler.shutdown()
        log.info("Done.")


if __name__ == "__main__":
    main()
