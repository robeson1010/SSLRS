from SSLRS.core import *
from fastai.vision.all import *
from fastai.distributed import *
import timm

df=pd.read_csv('./data/file.csv')
newdf=pd.concat([df[df['Isval']==0].sample(frac=0.05,random_state=10),df[df['Isval']==1]])
db = DataBlock(blocks=(TransformBlock(type_tfms=partial(MSTensorImage.create)), MultiCategoryBlock),
                   splitter=ColSplitter('Isval'),
                   get_x=ColReader('fname'),
                   get_y=ColReader('label', label_delim=' '),
                   item_tfms=[aug,aug2]
              )
dls = db.dataloaders(source=newdf, bs=256, num_workers=8)

model=xresnet18(c_in=10, n_out=19)
# model=timm.create_model('convmixer_768_32',num_classes=19, in_chans=10)
init_cnn(model)

learn = Learner(dls, model, metrics=partial(accuracy_multi, thresh=0.5)).to_fp16()

with learn.distrib_ctx(): 
    learn.fit_flat_cos(200, 1e-3,cbs=[CSVLogger(fname='0.05supres18.csv'),SaveModelCallback(monitor='accuracy_multi',fname='0.05supres18')])