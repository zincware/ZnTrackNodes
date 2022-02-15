import os
import shutil
import subprocess

import numpy as np
import pytest

from zntracknodes import helloworld


@pytest.fixture
def proj_path(tmp_path):
    shutil.copy(__file__, tmp_path)
    os.chdir(tmp_path)
    subprocess.check_call(["git", "init"])
    subprocess.check_call(["dvc", "init"])

    return tmp_path


def test_RandomNumber(proj_path):
    helloworld.RandomNumber(start=5, stop=50).write_graph(run=True)
    assert helloworld.RandomNumber.load().start == 5
    assert helloworld.RandomNumber.load().stop == 50
    assert helloworld.RandomNumber.load().number >= 5
    assert helloworld.RandomNumber.load().number <= 50


def test_get_random_number(proj_path):
    cfg = helloworld.get_random_number(run=True)
    assert int(cfg.outs.read_text()) >= 0
    assert int(cfg.outs.read_text()) < 10


def test_PrepareMnist(proj_path):
    helloworld.PrepareMnist().write_graph(run=True)
    mnist = helloworld.PrepareMnist.load()
    assert isinstance(mnist.x_test, np.ndarray)


def test_TrainMnistModel(proj_path):
    helloworld.PrepareMnist().write_graph(run=True)
    helloworld.TrainMnistModel().write_graph(run=True)

    assert helloworld.TrainMnistModel.load().model_metrics["accuracy"] > 0.8
