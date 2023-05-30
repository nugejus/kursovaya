import os
import numpy as np

def SaveTrainingLog(model,history,path_add=''):
    import json
    
    acc4file=str(round(history.history['val_accuracy'][-1],4))
    loss4file=str(round(history.history['val_loss'][-1],4))
    
    os.makedirs(path_add+'\\training_log\\'+f'acc_{acc4file}___loss_{loss4file}')
    path=path_add+'\\training_log\\'+f'acc_{acc4file}___loss_{loss4file}\\'+f'acc_{acc4file}___loss_{loss4file}'
    
    model.save_weights(path+'.h5')

    history_dict=history.history
    json.dump(history_dict, open(path+'.json', 'w'))
    
def SaveAndPlot(history,path_add=''):
    import matplotlib.pyplot as plt
    
    acc4file=str(round(history.history['val_accuracy'][-1],4))
    loss4file=str(round(history.history['val_loss'][-1],4))
    path=path_add+'\\training_log\\'+f'acc_{acc4file}___loss_{loss4file}\\'+f'acc_{acc4file}___loss_{loss4file}'
    
    # test_loss,test_acc=model3d.evaluate(X_test,y_test,verbose=2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.legend(['train','validation'])
    plt.savefig(path+'_loss'+'.png')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('accuracy')
    plt.legend(['train','validation'])
    plt.savefig(path+'_acc'+'.png')
    plt.show()
    

def x_FrameConc(x,w,h):
    max_frame=max([len(vid) for vid in x])
    X=[]
    for vid in x:
        if len(vid)<max_frame:
            padding_layer=np.zeros((max_frame-len(vid),w,h,3))
            X.append(np.concatenate((vid,padding_layer),axis=0))
        else:
            X.append(vid)
        
    return np.asarray(X),max_frame