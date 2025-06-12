import glob
import os
import pandas as pd
from collections import Counter
import numpy as np

def ensemble(dataset_name,num_classes,output_path):

    csvs =[]
    vote_dict = {}
    conf_dict = {}
    label_dict = {}
    percentage_dict = {}
    hard_results = []
    soft_results = []
    folders=os.listdir(output_path)
    for folder in folders:
        if glob.glob(f"{output_path}/{folder}/*_{dataset_name}_dataset.parquet"):
            csvs.append(glob.glob(f"{output_path}/{folder}/*_{dataset_name}_dataset.parquet").pop())
    for csv in csvs:
        df = pd.read_parquet(csv,engine='pyarrow')
        for i,row in df.iterrows():
            path = row['image_path']
            pred = row['prediction']
            conf = row['confidence']
            label = row['label']
            percentage = row['percentage']
            vote_dict.setdefault(path,[]).append(pred)
            conf_dict.setdefault(path,[]).append(conf)
            percentage_dict.setdefault(path,[]).append(percentage)
            label_dict[path] = label

    for path in vote_dict:
        preds = vote_dict[path]
        confs = conf_dict[path]
        percentages = percentage_dict[path]
        hard = Counter(preds).most_common(1)[0][0]
        
        probs = np.zeros(num_classes)
        for pred,percent in zip(preds,percentages):
            probs += percent
        soft = np.argmax(probs)
        hard_results.append({'image_path':path,'label':label_dict[path],'hard_pred':hard})
        soft_results.append({'image_path': path, 'label': label_dict[path], 'soft_pred': soft})

    hard_df = pd.DataFrame(hard_results)
    soft_df = pd.DataFrame(soft_results)

    soft_acc = len(soft_df[soft_df['label']==soft_df['soft_pred']])/len(soft_df) * 100
    hard_acc = len(hard_df[hard_df['label']==hard_df['hard_pred']])/len(hard_df) * 100

    pd.merge(hard_df,soft_df,on=['image_path','label']).to_csv(f"{output_path}/ensemble_{dataset_name}.csv", index=False)
    print(f"hard_enseble:{hard_acc:.2f},soft_ensemble_acc:{soft_acc:.2f},  saved at {output_path}/ensemble_{dataset_name}.csv")
