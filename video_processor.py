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
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') #save as mp4
    
    def VidConveter(self,path,issave):
        """
        VidConveter
        transforms video with size w(wigth),h(high),fps(frames)
        returns only figure of the pictures
        [issave=Bool] makes return value or saves as mp4 file
                      if issave=False then returns numpy array that 1 frame is 1 demension numpy array
        """
        #open video file from name
        capture=cv2.VideoCapture("raw_videos/"+path)
        if not capture.isOpened():
            # print('File open failed!')
            capture.release()
            return -1

        #open writer
        if(issave):
            writer=cv2.VideoWriter("cvted/"+"cvted_"+path.split('.')[0]+".mp4", self.fourcc, self.fps, (self.w, self.h))
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
            canny_frame=cv2.Canny(frame,40,60)#save only canny version
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
        


    def Json2Csv(self,file_name='',save_name='',save_file=True):
        """
        extracting videos as dataframe
        if save_file==True then saves as csv file
        """
        if file_name=='' or file_name.split('.')[-1]!='json':
            sys.exit('Expected .json file name to read')
        with open(file_name) as file:
            data = json.load(file)

        gloss2vid_num=dict()
        for j in range(len(data)):
            lst=[]
            for inst in data[j]['instances']:
                lst.append(inst['video_id'])
            gloss2vid_num[data[j]['gloss']]=lst

            
        #make dataframe column as the same lengtt
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
        

    def VideoReader(self,path,limit_class):
        """
        Reads video from directory "cvted/"
        returns x,y
        x contains 4dimentional array with 3 channel elements
        y contains 1dimentional array with string type elements
        """
        df_dict=pd.read_csv(path,dtype=str).to_dict()

        x=[]
        y=[]

        for key in df_dict.keys():
            if key==limit_class:
                break
            for index in df_dict[key].keys():
                if df_dict[key][index]=='-1':
                    continue
                path1="cvted_"+str(df_dict[key][index])+".mp4"
                capture=cv2.VideoCapture("cvted/"+path1)
                if not capture.isOpened():
                    capture.release()
                    sys.exit('File open failed')
                frames=[]
                while(True):
                    retval,frame=capture.read()
                    if not retval:
                        break
                    frames.append(frame)
                x.append(frames)
                y.append(key)
                
        return x,y
    
    
    def CsvCorrector(self,path,file_name,save_name):
        """
        Compares csv file with directory file list
        If name of video is not in directory then replaces file name in csv as -1
        returns nothing
        """
        import glob
        import re
        df=self.Json2Csv(file_name=file_name,save_file=False)
        dir_list=glob.glob(path) #path requires directory path of not converted videos
        res=[re.search(r'\b\d{5}\b',dir_list[i]).group(0) for i in range(len(dir_list))]

        df_dict=df.to_dict()
        for key in df_dict.keys():
            for num in df_dict[key].keys():
                if df_dict[key][num] not in res:
                    df_dict[key][num]=-1

        pand=pd.DataFrame(df_dict)
        pand.to_csv(save_name,index=False)
