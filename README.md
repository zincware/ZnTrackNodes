# Collection of ZnTrack Nodes

- TensorFlow Example on the MNIST Dataset
````python
from zntracknodes.helloworld import PrepareMnist, TrainMnistModel

PrepareMnist().write_graph(run=True)
TrainMnistModel(epochs=7).write_graph(run=True)
print(TrainMnistModel.load().model_metrics)
# {"accuracy": 0.97562, "loss": 0.064564}
````