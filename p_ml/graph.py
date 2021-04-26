import seaborn as sn
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show 
from bokeh.palettes import magma
from bokeh.models import ColorBar, ColumnDataSource
from bokeh.palettes import Viridis6
from bokeh.plotting import figure, output_file, show
from bokeh.transform import linear_cmap

        


class Graph:


    def confusion_matrix(self, data):
        confusion_matrix = pd.crosstab(data[0], data[1], rownames=['Actual'], colnames=['Predicted'])
        sn.heatmap(confusion_matrix, annot=True)
        plt.show()
        

    def significant_features(self, data):
        output_file("significant_features.html") 
        graph = figure(title = "Significant Gene Features") 
        graph.xaxis.axis_label = "Gene"
        graph.yaxis.axis_label = "Feature Importance"
        y_whole = data.index.values
        y = [z for z in y_whole if 0.001 <= z <= 2]
        x = list(range(2, len(y)))
        x = x.reverse()
        size = 40
        mapper = linear_cmap(field_name='y', palette=Viridis6 ,low=min(y) ,high=max(y))
        graph.scatter(x, y, size = size, color = mapper) 
        show(graph)
