import numpy as np

from ivory.lightgbm.estimator import Classifier


def test_lgbr(client):
    experiment = client.create_experiment("lgbr")
    run = experiment.create_run()
    run.start()


def test_lgbc():
    classifier = Classifier(num_boost_round=3, num_class=10)
    train_x = np.random.rand(100, 20)
    train_y = np.random.randint(0, 10, 100)
    val_x = np.random.rand(100, 20)
    val_y = np.random.randint(0, 10, 100)
    classifier.fit([train_x, train_y], [val_x, val_y])
    assert classifier.predict(val_x).shape == (100, 10)
