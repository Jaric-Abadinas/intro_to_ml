import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(url, header=None, names=column_names)

#Save the DataFrame to a local CSV file
df.to_csv("iris.csv", index=False)
df = pd.read_csv("iris.csv")