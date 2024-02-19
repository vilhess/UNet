import os
import torch
from tqdm import tqdm

def get_F1(unet, dataset, device='cpu'):
  tp_count = 0
  tn_count = 0
  fp_count = 0
  fn_count = 0
  with torch.no_grad():
    for i, data in enumerate(dataset):
      image = data[0]
      mask = data[1]
      h,w = mask.size()
      nh = (h//32)*32
      nw = (w//32)*32
      image = image[:,:nh,:nw].to(device)
      mask = mask[:nh,:nw].to(device)
      out = unet(torch.unsqueeze(image,0))
      predmask = torch.argmax(out.squeeze(), dim=0)
      # count:
      tp = torch.sum(predmask*mask)
      fp = torch.sum(predmask*(1.-mask))
      tn = torch.sum((1-predmask)*(1.-mask))
      fn = torch.sum((1-predmask)*mask)
      if not (tp+fp+tn+fn == nh*nw):
        print('counting inconstancy')
      tp_count = tp_count + tp
      tn_count = tn_count + tn
      fp_count = fp_count + fp
      fn_count = fn_count + fn

    precision = (tp_count/(tp_count+fp_count)).to('cpu').item()
    recall = (tp_count/(tp_count+fn_count)).to('cpu').item()
    F1score =  2. * precision * recall / (precision + recall)
    return F1score
  

def test_performance(unet, dataset, device='cpu'):
  tp_count = 0
  tn_count = 0
  fp_count = 0
  fn_count = 0
  with torch.no_grad():
    for data in tqdm(dataset):
      image = data[0]
      mask = data[1]
      h,w = mask.size()
      nh = (h//32)*32
      nw = (w//32)*32
      image = image[:,:nh,:nw].to(device)
      mask = mask[:nh,:nw].to(device)
      out = unet(torch.unsqueeze(image,0))
      predmask = torch.argmax(out.squeeze(), dim=0)
      # count:
      tp = torch.sum(predmask*mask)
      fp = torch.sum(predmask*(1.-mask))
      tn = torch.sum((1-predmask)*(1.-mask))
      fn = torch.sum((1-predmask)*mask)
      if not (tp+fp+tn+fn == nh*nw):
        print('counting inconstancy')
      tp_count = tp_count + tp
      tn_count = tn_count + tn
      fp_count = fp_count + fp
      fn_count = fn_count + fn

    precision = (tp_count/(tp_count+fp_count)).to('cpu').item()
    recall = (tp_count/(tp_count+fn_count)).to('cpu').item()
    F1score =  2. * precision * recall / (precision + recall)

    valscores = [precision, recall, F1score]

    for name, val in zip(["precision", "recall", "F1score"], valscores):
        print(name, '{:.4f}'.format(val))