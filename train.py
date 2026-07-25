import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-mpcafsa.yaml')    #ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml
    # model.load('') # loading pretrain weights
    model.train(data='D:/long(rt-detr)/RTDETR/RTDETR-main/dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0,1',
                # resume='', # last.pt path
                project='runs/train',
                name='rtdetr-mpcafsa(TT100K)',
                )