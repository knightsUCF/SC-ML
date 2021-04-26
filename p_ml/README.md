# SC-ML


## Data

Data was obtained from the Actinn single cell prediction project. All of the data was used for the 1000 available labeled names. A 50 / 50, train test split was used.

https://github.com/mafeiyang/ACTINN

## Running

To run the models run main.py:

```
import data
import net
import forest


data = data.data()
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
```

Output of significant features which subset immune system cells by genes:


