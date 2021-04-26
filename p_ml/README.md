# SC-ML


## Data

Data was obtained from the Actinn single cell prediction project. A 50 / 50, train test split was used.

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


```
Run 1

             Gene names
0.022858          H2-Aa
0.020494         H2-Eb1
0.019538         H2-Ab1
0.017683           Cd74
0.015444          Ms4a1
0.012397  2010001M09Rik
0.012198          Faim3
0.011395          Cd79a
0.010870           Cd3d
0.010579          Cd79b

Run 2

         Gene names
0.021301       Cd3e
0.019088       Cd74
0.017432     H2-Ab1
0.015671      Ms4a1
0.014209     H2-Eb1
0.014091      Cd79b
0.012021       Cd3d
0.010348       Cd3g
0.010136      Cd79a
0.009748      H2-Aa

Run 3

         Gene names
0.018718      Ms4a1
0.017804      H2-Aa
0.016925     H2-Eb1
0.015720      Faim3
0.014539     H2-Ab1
0.013759       Cd3d
0.011119      Cd79b
0.009415       Cd3g
0.009356      H2-Ob
0.009338       Cd3e

Run 4

         Gene names
0.025916       Cd3d
0.024558      Cd79a
0.023257       Cd74
0.014324     H2-Ab1
0.014192       Cd3g
0.013893     H2-Eb1
0.012758      Skap1
0.011806      Cd8b1
0.011402       Thy1
0.010209      H2-Aa

Run 5

         Gene names
0.020163      Cd79a
0.019039     H2-Ab1
0.018109      Cd79b
0.016304       Cd74
0.014754      Bank1
0.013799      Ms4a1
0.013105       Thy1
0.012642       Cd19
0.011194     Tmem66
0.010675       Cd3d

Run 6

         Gene names
0.031415       Cd74
0.019561       Cd3d
0.018283        Lck
0.014832      Napsa
0.014694       Cd3e
0.012517     Tmem66
0.011640     H2-Ab1
0.011616      Ms4a1
0.010861     H2-Eb1
0.010770       Tcf7


Run 7

         Gene names
0.025716     H2-Eb1
0.018039      Cd79a
0.016733       Cd3g
0.015565       Thy1
0.015032      H2-Aa
0.013606      Ms4a1
0.012722        Lck
0.012422       Cd3d
0.012372     H2-Ab1
0.011443      Faim3

Run 8

0.016903           Cd3d
0.016701          H2-Aa
0.014987           Cd74
0.013875          Cd79b
0.012461         Tyrobp
0.011850          Skap1
0.011821            Lat
0.011232  2010001M09Rik
0.011051            Fyb
0.010734          Napsa

Run 9

         Gene names
0.023840      H2-Aa
0.023666      Cd79a
0.022787       Cd74
0.018044       Cd3g
0.016889     H2-Eb1
0.014441        Lck
0.011481      Ms4a1
0.009944       Cd3d
0.009869       Cd3e
0.009827       Thy1

Run 10
         Gene names
0.023075      H2-Aa
0.019307     H2-Eb1
0.018554     H2-Ab1
0.018345      Cd79b
0.016545       Cd3e
0.015942       Ly6d
0.015445       Cd3d
0.014053       Cd3g
0.012847      Cd79a
0.012444      Ms4a1
```

## Confusion Matrix

Note: this confusion matrix was run at a different time than the above data, so this is within 1% of variance. However the charts below were run at the same time as the above data.

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/confusion%20matrix.png)

## Run 1

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/run%201.png)

## Run 2

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/run%202.png)

## Run 3

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/run%203.png)

## Run 4

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/run%204.png)

## Run 5

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/run%205.png)

## Run 6

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/run%206.png)


## Run 7

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/run%207.png)

## Run 8

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/run%208.png)

## Run 9

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/run%209.png)

## Run 10

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/run%2010.png)
