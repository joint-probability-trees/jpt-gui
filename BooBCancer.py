import sklearn.datasets

data = sklearn.datasets.load_breast_cancer(as_frame=True)

df = data.data
target = data.target
target[target == 1] = "malignant"
target[target == 0] = "friendly"

df["malignant"] = target
import jpt.trees
import jpt.variables

variables = jpt.variables.infer_from_dataframe(df)

model = jpt.trees.JPT(variables, min_samples_leaf=1/569)

model.fit(df)

model.save("boob_cancer.jpt")
