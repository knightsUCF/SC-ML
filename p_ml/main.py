import data
import net
import forest
import graph


data = data.Data()
net = net.Net()
forest = forest.Forest()
graph = graph.Graph()



features = data.get_features()
targets = data.get_targets()


# SuperCT
net.build()
net.train(features, targets)
net.evaluate()


# Random Forest
forest.build()
forest.train(features, targets)
forest.evaluate()
