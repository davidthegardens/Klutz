import json
import pandas as pd
import polars as pl
from genealogy import Gene
import numpy as np
import web3

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
    uniquelabels=df['labels'].to_list()
    with open("/Users/davidthegardens/Documents/python/Klutz/etherscan-labels/data/etherscan/combined/combinedTokenLabels.json") as json_file:
        data=json.load(json_file)

    labeldict={}
    addrList=[]

    for i in data:
        label_list=data[i]['labels']
        for j in label_list:
            print(i)
            if j in uniquelabels:
                
                if j in labeldict.keys():
                    labeldict[j].append(i)
                else: 
                    labeldict[j]=[i]
    print(labeldict)

#getAddresses()

def cleanLlama():
    cleaned_data={'name':[],'category':[],'address':[]}
    with open("llama.json") as json_file:
        data=json.load(json_file)
    for iter in data:
        if 'Ethereum' in iter['chains'] and iter['address']!=None:
            cleaned_data['name'].append(iter['name'])
            cleaned_data['category'].append(iter['category'])
            cleaned_data['address'].append(iter['address'])
    df=pl.DataFrame(cleaned_data)
    df.write_parquet('cleaned_data.parquet')

#cleanLlama()
def getFamily():
    fam_list=[]
    g=Gene()
    df=pl.read_parquet('cleaned_data.parquet')
    addr_list=df['address'].to_list()
    counter=1
    try:
        finaldf=pl.read_parquet('data_with_fam.parquet')
        null_count=finaldf['family'].null_count()
        dfx=df.tail(null_count)
        addr_list=dfx['address'].to_list()
        completed_rows=(finaldf.height-null_count)
        counter=completed_rows+1
        fam_list=finaldf.head(completed_rows)['family'].to_list()
    except Exception:
        pass

    for addr in addr_list:
        if addr[:2]!="0x":
            if addr[:4]=="eth:":
                addr=addr.lstrip('eth:')
            else:
                continue
            continue
        print("fetching "+addr)
        try:
            fam_list.append(g.fastFamily(addr,50))
        except Exception:
            return "unfinished"
        df2=pl.DataFrame({'family':fam_list})
        df2.fill_null([])
        finaldf=pl.concat([df,df2],how='horizontal')
        #df.with_columns(pl.Series(name="family", values=fam_list)) 
        finaldf.write_parquet('data_with_fam.parquet')
        print(finaldf.head(counter))
        counter+=1
    return "finished"

def getFamily_recursive():
    status='unfinished'        
    while status!='finished':
        status=getFamily()

