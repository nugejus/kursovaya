import json
import pandas as pd

def extract_glosses(file_name,save_name='',save_file=True):
    with open(file_name) as file:
        data = json.load(file)

    gloss2vid_num=dict()
    for j in range(len(data)):
        lst=[]
        for inst in data[j]['instances']:
            lst.append(inst['video_id'])
        gloss2vid_num[data[j]['gloss']]=lst

    max_range=max([len(x) for x in gloss2vid_num.values()])

    for key in gloss2vid_num.keys():
        for i in range(len(gloss2vid_num[key]),max_range):
            gloss2vid_num[key].append('-1')


    data_frame=pd.DataFrame(gloss2vid_num)
    
    if(save_file):
        data_frame.to_csv(save_name,index=False)
    else:
        return data_frame