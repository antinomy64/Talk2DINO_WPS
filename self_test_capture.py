import os,tempfile,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
from ours_val_triptych_capture import Capture
class Dataset:
    def __init__(self,folder):
        self.img_infos=[]
        for i in range(2):
            p=folder/f'test_{i:03}.png'
            Image.fromarray(np.full((6,7,3),150,dtype=np.uint8)).save(p)
            self.img_infos.append({'filename':str(p)})
        self.img_dir=str(folder)
    def get_gt_seg_map_by_idx(self,idx):
        gt=np.full((6,7),2,dtype=np.uint8)
        gt[0,0]=255
        gt[1,1]=3
        return gt
class Model:
    def forward_test(self,imgs,img_metas):
        return [np.full((6,7),4,dtype=np.uint8)]
with tempfile.TemporaryDirectory() as d:
    d=Path(d)
    os.environ.update(OVPS_VIS_DATASET='pp116',OVPS_VIS_EXPECTED_COUNT='2',OVPS_VIS_EXPECTED_MIOU='36.75',OVPS_VIS_OUTPUT=str(d/'output'))
    ds=Dataset(d)
    model=Model()
    obj=Capture(model,ds)
    for info in ds.img_infos:
        x=model.forward_test(None,[[{'ori_filename':info['filename'],'filename':info['filename']} ]])
        assert np.asarray(x[0]).shape==(6,7)
    obj.finish(36.75)
    pngs=list((d/'output'/'triptychs').glob('*.png'))
    assert len(pngs)==2
    arr=np.asarray(Image.open(pngs[0]))
    assert arr.shape==(6+35,3*7+2*8,3),arr.shape
    assert (arr[35,7+8, :] == [150,150,150]).all() # GT ignore
    print('[MOCK END-TO-END PASS]',len(pngs),'triptychs, original GT ignore preserved')
