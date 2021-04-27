import data
import net
import forest
import graph
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


data = data.Data()
forest = forest.Forest()

features = data.get_features()
targets = data.get_targets()



runs = 1000 # how many times to run random forest before extracting averages
samples_to_graph = 100


# Random Forest
forest.build()
forest.train(features, targets)




def determine_averages(runs):
    # we need to build a reference dataframe, which will contain the gene names as indexes to merge
    df0 = forest.get_significant_features()
    df0['significance'] = df0.index
    df0 = df0.set_index('Gene names')

    for i in range(runs):
        print('run ', i)
        forest.train(features, targets)
        df1 = forest.get_significant_features()
        df1['significance'] = df1.index
        df1 = df1.set_index('Gene names')
        df0 = df0.merge(df1, left_index=True, right_index=True)

    df0['average'] = df0.mean(numeric_only=True, axis=1)
    df0['rank'] = df0['average'].rank(ascending = 0)
    df0 = df0.sort_values(by = 'rank')

    return df0



significant_genes = determine_averages(runs)


fig1 = px.scatter(significant_genes[:samples_to_graph], x = significant_genes.index[:samples_to_graph], y=significant_genes['average'][:samples_to_graph]) 

fig1.update_layout(
    title="Significant Immune System Cell Specific Genes",
    xaxis_title="Gene",
    yaxis_title="Significance",
    legend_title="Legend Title",
    font=dict(
        family="Helvetica, monospace",
        size=15,
        color="RebeccaPurple"
    )
)

fig1.show()


