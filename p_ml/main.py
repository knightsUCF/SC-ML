import data
import forest
import graph


data = data.data()
forest = forest.Forest()
graph = graph.Graph()


features = data.get_features()
targets = data.get_targets()


# SuperCT
net.build()
net.train(features, targets)
net.evaluate()


# Random Forest improvement
forest.build()
forest.train(features, targets)
forest.evaluate()
