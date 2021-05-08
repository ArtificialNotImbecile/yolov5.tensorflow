import torch
import pickle
import sys
import numpy as np

# 填写对应路径
sys.path.append('/notebooks/experimental/yolov5')
pt_file = '/notebooks/experimental/yolov5/weights/yolov5m.pt'

device = torch.device("cpu")
model = torch.load(pt_file, map_location=device)
model = model['model'].float()  # load to FP32
model.to(device)
model.eval()

meta = {'nc': model.nc, 'strides': model.stride.numpy().astype(np.int64).tolist(),
        'anchors': model.yaml['anchors'], 'model_name': model.yaml_file[6]}
with open('./meta.dict', 'wb') as f:
    pickle.dump(meta, f)

data_dict = {}
for k, v in model.state_dict().items():
    vr = v.cpu().numpy()
    data_dict[k] = vr

with open('params.dict', 'wb') as f:
    pickle.dump(data_dict, f)
