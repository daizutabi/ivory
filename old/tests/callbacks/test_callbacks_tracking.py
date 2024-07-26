def test_no_metrics(run):
    tracking = run.tracking
    metrics = run.dict.pop("metrics")
    tracking.on_epoch_end(run)
    run.set(metrics=metrics)
