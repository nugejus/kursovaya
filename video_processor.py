import cv2
import sys
import numpy as np
import json
import pandas as pd

class process:
    def __init__(self,w,h,fps):
        self.w=w
        self.h=h
        self.fps=fps
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') #mp4형식으로 저장
    
    """
    vid_proc
    transforms video with size w(wigth),h(high),fps(frames)
    returns only figure of the pictures
    [issave=Bool] makes return value or saves as mp4 file
                  if issave=False then returns numpy array that 1 frame is 1 demension numpy array
    """
    def vid_proc(self,path,issave):
        #open video file from name
        capture=cv2.VideoCapture("raw_videos/"+path)
        if not capture.isOpened():
            # print('File open failed!')
            capture.release()
            return -1

        #open writer
        if(issave):
            writer=cv2.VideoWriter("cvted/"+"cvted_"+path+".mp4", self.fourcc, self.fps, (self.w, self.h))
            if not writer.isOpened():#if failed to open video writer
                print('Writer open failed!')
                writer.release()
                capture.release()
                sys.exit()
        else:
            return_vec=[]

        while(True):
            retval, frame = capture.read()
            if not retval:
                break;
            canny_frame=cv2.Canny(frame,40,60)#윤곽선만 저장
            res_frame=cv2.resize(canny_frame,(self.h,self.w),interpolation=cv2.INTER_AREA)
            edge_color = cv2.cvtColor(res_frame, cv2.COLOR_GRAY2BGR) #because canny object doesn't saves
            if(issave):
                writer.write(edge_color)
            else:
                return_vec.append(edge_color)
                
        # closing
        capture.release()
        if(issave):
            writer.release()
        else:
            res=[]
            for frame in return_vec:
                mean=[]
                for vec in frame:
                    mean+=list(np.mean(vec,axis=1))
                res.append(mean)
            return np.array(res)
        


    def extract_glosses(self,file_name='',save_name='',save_file=True):
        if file_name=='':
            print('Expected .json file name to read')
            sys.exit()
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
            return 1
        else:
            return data_frame