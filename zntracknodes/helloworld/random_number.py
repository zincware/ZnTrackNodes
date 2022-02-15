import pathlib
import random

from zntrack import Node, NodeConfig, nodify, zn


class RandomNumber(Node):
    """Compute a Random Number on the Graph

    Attributes
    ----------
    start: int, default = 0
        start value for randrange
    stop: int
        stop value for randrange
    step: int, default = 1
        step value for randrange
    number: int
        The computed random number
    """

    start: int = zn.params(0)
    stop: int = zn.params()
    step: int = zn.params(1)
    number: int = zn.outs()

    def run(self):
        """Compute a random number"""
        self.number = random.randrange(start=self.start, stop=self.stop, step=self.step)


@nodify(params={"start": 0, "stop": 10, "step": 1}, outs=pathlib.Path("number.txt"))
def get_random_number(cfg: NodeConfig):
    """Compute a Random Number on the Graph"""
    number = random.randrange(
        start=cfg.params.start, stop=cfg.params.stop, step=cfg.params.step
    )
    cfg.outs.write_text(f"{number}")
