# Legacy daily-bot deployment

These systemd files schedule the legacy daily bot and IB Gateway; they do not launch the Kafka event-driven services.

Use the root README and [eventbot operations](../docs/operations.md) for the event-driven stack. The legacy files remain for reference and shared strategy compatibility. Their installation and unattended operation were not validated during publication.

## Manual setup

Review `install.sh` and the unit files before using them. They expect a user-level systemd environment, a suitable Python environment, IB Gateway, IBC and a virtual display. Account credentials belong in IBC's private configuration outside this repository. Keep the Gateway configured for paper trading.

The installer can change a user timer and Gateway service, so installation is a separate operator action. It can also conflict with an existing cron job or Gateway session. A `DRYRUN` file controls the legacy wrapper; removing it can enable paper execution on a scheduled run.

## Limitations

No service installation, scheduling change, Gateway login or order was performed as part of cleanup. The legacy runner has different controls from eventbot's strict paper guard and includes an explicit live opt-in. It should not be treated as the documented event-driven entry point. Do not infer live-readiness from the offline tests or from an enabled timer.
