import json
import pandas as pd

def getUnique():
    with open("/Users/davidthegardens/Documents/python/Klutz/etherscan-labels/data/etherscan/combined/combinedTokenLabels.json") as json_file:
        data=json.load(json_file)

    unique_labels=[]

    for i in data:
        label_list=data[i]['labels']
        for j in label_list:
            if j not in unique_labels:
                unique_labels.append(j)

    df=pd.DataFrame({"labels":unique_labels})
    print(df)
    df.to_csv("uniquelabels.csv")

def getAddresses():
    df=pd.read_csv("uniquelabels.csv")
    print(df)
    # uniquelabels=df['labels'].to_list()
    # with open("/Users/davidthegardens/Documents/python/Klutz/etherscan-labels/data/etherscan/combined/combinedTokenLabels.json") as json_file:
    #     data=json.load(json_file)

    # labeldict={}

    # for i in data:
    #     label_list=data[i]['labels']
    #     for j in label_list:
    #         if j in uniquelabels:
    #             if j in labeldict.keys():
    #                 print(labeldict[j])
    #                 labeldict[j]=labeldict[j].append(i)
    #         else:
    #             labeldict[j]=[i]
    # print(labeldict)

getAddresses()