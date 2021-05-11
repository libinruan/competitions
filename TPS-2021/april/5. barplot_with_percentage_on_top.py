# SOURCE https://www.kaggle.com/c/tabular-playground-series-apr-2021/discussion/230488

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# step 1.

def fare_split(fare_num):
    if fare_num < 10:
        return 1
    elif fare_num < 25:
        return 2 
    elif fare_num < 60:
        return 3 
    elif fare_num < 100:
        return 4
    elif fare_num < 375: 
        return 5

df =  pd.read_csv("../../data/train.csv")

df["Fare_range"] = df["Fare"].fillna(0).apply(fare_split)             

# step 2.

def describe_category(dataframe, column_name):
    """
    plot describe category with percentage
    """
    value_count = dataframe[column_name].value_counts().sort_index()
    df_value_count = pd.DataFrame({column_name: value_count.index, "count": value_count.values})
    sum_class = df_value_count["count"].sum()
    df_value_count["percentage"] = df_value_count["count"]/sum_class*100
    display(df_value_count)

    # fig, ax = plt.subplots(figsize = (1,7)
    ax = sns.barplot(data=df_value_count, x=column_name, y="count")
    ax.set_ylim(0, df_value_count["count"].max()*1.2)
    for p, percentage in zip(ax.patches, list(df_value_count["percentage"])):
        ax.annotate("%.2f" % percentage +" %", (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', rotation=0, xytext=(0, 20), textcoords='offset points')  #vertical bars

describe_category(df, 'Fare_range')