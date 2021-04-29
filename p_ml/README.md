# SC-ML

### Prediction Model Improvement of Cell Types by Gene Expression

Here is the overview for the project which aimed to improve upon the SuperCT method of classifying cells based on gene expression.


## Contents

#### I. Data
#### II. Running
#### III. Confusion Matrix
#### IV. Charts for Gene Significant Runs
#### V. Gene Significance Averaged After 1000 Random Forest Runs
#### VI. Most Significant Gene Features per Immune Cell Type



## I. Data

Data was obtained from the Actinn single cell prediction project. A 50 / 50, train test split was used.

https://github.com/mafeiyang/ACTINN

<b>Features - Genes (14,000) / Samples - Barcode Cell IDs (1,000) </b>

```
               0610005C13Rik  0610007C21Rik  0610007L01Rik  0610007N19Rik  0610007P08Rik  ...  Zyx  Zzef1  Zzz3    a  l7Rn6
tma_mfd_1808             0.0            0.0            1.0            0.0            0.0  ...  0.0    0.0   1.0  0.0    0.0
tma_mfd_32608            0.0            0.0            0.0            0.0            0.0  ...  0.0    0.0   0.0  0.0    0.0
tma_mfd_2589             0.0            0.0            1.0            0.0            0.0  ...  0.0    0.0   0.0  0.0    0.0
tma_mfd_13999            0.0            0.0            2.0            0.0            0.0  ...  1.0    1.0   0.0  0.0    0.0
tma_mfd_2621             0.0            0.0            0.0            0.0            0.0  ...  0.0    0.0   0.0  0.0    1.0
...                      ...            ...            ...            ...            ...  ...  ...    ...   ...  ...    ...
tma_mfd_2459             0.0            0.0            0.0            0.0            0.0  ...  1.0    0.0   0.0  0.0    1.0
tma_mfd_13712            0.0            0.0            0.0            0.0            0.0  ...  1.0    0.0   0.0  0.0    0.0
tma_mfd_33269            0.0            1.0            0.0            0.0            0.0  ...  1.0    0.0   0.0  0.0    0.0
tma_mfd_32996            0.0            0.0            1.0            1.0            0.0  ...  0.0    0.0   1.0  0.0    3.0
tma_mfd_33381            0.0            1.0            0.0            0.0            0.0  ...  0.0    0.0   0.0  0.0    0.0
[1000 rows x 14063 columns]
```

<b>Labels aka Targets - Cell Types (1,000)</b>

```
                 0            1
0     tma_mfd_1808       B cell
1    tma_mfd_32608       T cell
2     tma_mfd_2589       B cell
3    tma_mfd_13999  Granulocyte
4     tma_mfd_2621       B cell
..             ...          ...
995   tma_mfd_2459       B cell
996  tma_mfd_13712  Granulocyte
997  tma_mfd_33269       T cell
998  tma_mfd_32996       T cell
999  tma_mfd_33381       T cell
```


## II. Running

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

## III. Confusion Matrix

Note: this confusion matrix was run at a different time than the above data, so this is within a 1% variance. However the charts below were run at the same time as the above data.

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/confusion%20matrix.png)

## IV. Charts for Gene Significant Runs

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/trials.png)

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


# V. Gene Significance Averaged After 1000 Random Forest Runs

![](https://github.com/knightsUCF/SC-ML/blob/main/p_ml/images/Gene%20Significance%201000%20Runs.png)


# VI. Most Significant Gene Features per Immune Cell Type

```Python
import data

data = data.Data()

data.get_gene_counts_per_cell('B cell')
data.get_gene_counts_per_cell('T cell')
data.get_gene_counts_per_cell('NK cell')
data.get_gene_counts_per_cell('Granulocyte')
data.get_gene_counts_per_cell('Monocyte')
```

```
running counts per cell type:  B cell

Most significant gene counts:

H2-Aa
H2-Eb1
Cd79a

Cd79a:  5714.0
H2-Aa:  9838.0
H2-Eb1:  7507.0
Cd3d:  29.0
Cd3e:  6.0
Faim3:  1491.0
Tyrobp:  208.0
Fyb:  10.0
Fcer1g:  44.0
Tmem66:  75.0
2010001M09Rik:  1418.0
Mef2c:  743.0
Fxyd5:  227.0
Skap1:  8.0
Cd247:  3.0
Gzma:  17.0
Nkg7:  17.0


running counts per cell type:  T cell

Most significant gene counts:

Cd3d
Tmem66
Cd3e

Cd79a:  29.0
H2-Aa:  100.0
H2-Eb1:  75.0
Cd3d:  1768.0
Cd3e:  904.0
Faim3:  6.0
Tyrobp:  35.0
Fyb:  511.0
Fcer1g:  60.0
Tmem66:  929.0
2010001M09Rik:  17.0
Mef2c:  2.0
Fxyd5:  885.0
Skap1:  440.0
Cd247:  411.0
Gzma:  88.0
Nkg7:  551.0


running counts per cell type:  NK cell

Most significant gene counts:

Gzma
Nkg7
Tyrobp


Cd79a:  30.0
H2-Aa:  80.0
H2-Eb1:  55.0
Cd3d:  12.0
Cd3e:  0.0
Faim3:  19.0
Tyrobp:  1083.0
Fyb:  46.0
Fcer1g:  817.0
Tmem66:  54.0
2010001M09Rik:  21.0
Mef2c:  12.0
Fxyd5:  225.0
Skap1:  49.0
Cd247:  36.0
Gzma:  3609.0
Nkg7:  1439.0


running counts per cell type:  Granulocyte

Most significant gene counts:

Tyrobp
Fcer1g
Fxyd5

Cd79a:  16.0
H2-Aa:  18.0
H2-Eb1:  13.0
Cd3d:  11.0
Cd3e:  0.0
Faim3:  1.0
Tyrobp:  1571.0
Fyb:  32.0
Fcer1g:  1151.0
Tmem66:  12.0
2010001M09Rik:  4.0
Mef2c:  4.0
Fxyd5:  229.0
Skap1:  0.0
Cd247:  0.0
Gzma:  1.0
Nkg7:  97.0


running counts per cell type:  Monocyte

Most significant gene counts:

Tyrobp
Fcer1g
Fxyd5

Cd79a:  8.0
H2-Aa:  233.0
H2-Eb1:  138.0
Cd3d:  2.0
Cd3e:  0.0
Faim3:  2.0
Tyrobp:  2083.0
Fyb:  112.0
Fcer1g:  1565.0
Tmem66:  48.0
2010001M09Rik:  2.0
Mef2c:  18.0
Fxyd5:  615.0
Skap1:  0.0
Cd247:  0.0
Gzma:  22.0
Nkg7:  8.0
```
