import data
import net
import forest
import graph
import pandas as pd


data = data.Data()
forest = forest.Forest()


features = data.get_features()
targets = data.get_targets()


# Random Forest
forest.build()
forest.train(features, targets)

# we need to build a reference dataframe, which will gave the gene names as indexes to merge

df0 = forest.get_significant_features()
df0['significance'] = df0.index
df0 = df0.set_index('Gene names')


for i in range(2):
    forest.train(features, targets)
    df1 = forest.get_significant_features()
    df1['significance'] = df1.index
    df1 = df1.set_index('Gene names')
    df0 = df0.merge(df1, left_index=True, right_index=True)


df0['average'] = df0.mean(numeric_only=True, axis=1)
df0['rank'] = df0['average'].rank(ascending = 0)
df0 = df0.sort_values(by = 'rank')

print(df0)
