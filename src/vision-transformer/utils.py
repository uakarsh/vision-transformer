import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
import json
import pandas as pd
from sklearn.metrics import accuracy_score as accuracy_score

class Average(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        
    @property
    def avg(self):
        return (self.sum / self.count) if self.count>0 else 0


class Logger:
    def __init__(self,filename,format='csv'):
        self.filename = filename + '.' + format
        self._log = []
        self.format = format

    def save(self,log,epoch=None):
        log['epoch'] = epoch+1
        self._log.append(log)
        if self.format == 'json':
            with open(self.filename,'w') as f:
                json.dump(self._log,f)
        else:
            pd.DataFrame(self._log).to_csv(self.filename,index=False)

def train_fn(data_loader,model,criterion,optimizer,device):
    model.train()
    
    tk0 = tqdm(data_loader, total=len(data_loader),leave=False)
    log = None
    
    for step, (images, targets) in enumerate(tk0):
        
        batch_size = len(images)
        images = images.to(device)
        targets = targets.to(device)

        output = model(images)
        loss = criterion(output, targets)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Updating the loss in a list
        if log is None:
            log = {'loss': Average()}
            log['accuracy'] = Average()



        log['loss'].update(loss.item(),batch_size)

        
        y_true = targets.detach().cpu().numpy()
        y_pred =  output.argmax(axis = -1).detach().cpu().numpy()
        accuracy = accuracy_score(y_pred,y_true)
        log['accuracy'].update(accuracy,batch_size)

        tk0.set_postfix({k:v.avg for k,v in log.items()}) 
        
    return log

def eval_fn(data_loader, model,criterion, device):

    model.eval()
    log = None
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader),leave=False)
        for step, (images,targets) in enumerate(tk0):
            
            batch_size = len(images)
            images = images.to(device)
            targets = targets.to(device)
            
            output = model(images)
            loss = criterion(output, targets)

            if log is None:
                log = {'loss': Average()}
                log['accuracy'] = Average()

            log['loss'].update(loss.item(),batch_size)
            y_true = targets.detach().cpu().numpy()
            y_pred =  output.softmax(axis = -1).argmax(axis = -1).detach().cpu().numpy()
            accuracy = accuracy_score(y_pred,y_true)
            log['accuracy'].update(accuracy,batch_size)
            
            
            tk0.set_postfix({k:v.avg for k,v in log.items()}) 
    
    return log 
            
            
def run_network(model,train_ds,val_ds, name, lr = 1e-3 , base_path = '',epochs=5, batch_size = 8):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = Logger(f'logs_{name}')
    train_dataloader = DataLoader(train_ds,batch_size = batch_size, shuffle = False)
    val_dataloader = DataLoader(val_ds, batch_size = batch_size, shuffle = False)

    
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    header_printed = False
    best_loss = 10**5
    for epoch in range(epochs):
      train_log = train_fn(train_dataloader, model,criterion, optimizer,device )
      valid_log = eval_fn(val_dataloader, model,criterion, device)

      log = {k:v.avg for k,v in train_log.items()}
      log.update({'V/'+k:v.avg for k,v in valid_log.items()})
      logger.save(log,epoch)
      keys = sorted(log.keys())
        
      if not header_printed:
            print(' '.join(map(lambda k: f'{k[:8]:8}',keys)))
            header_printed = True
      print(' '.join(map(lambda k: f'{log[k]:8.3f}'[:8],keys)))
        
      if log['V/loss'] < best_loss:
            best_ce = log['V/loss']
            print('Best model found at epoch {}'.format(epoch+1))
            save_path = os.path.join(base_path, f'{name}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)


    return logger


class ImageNetDataset(Dataset):
    
    def __init__(self, image_entry, label_entry, resize_shape = (224,224), patch_size = (16,16), transform = None):
        self.image_entry = image_entry
        self.label_entry = label_entry
        self.resize_shape = resize_shape

        self.x_patch = patch_size[0]
        self.y_patch = patch_size[1]
        self.transform = transform

    def __len__(self):
        return len(self.image_entry)

    def __getitem__(self, idx):

        entry = self.image_entry[idx]
        label = torch.as_tensor(self.label_entry[idx])

        img = Image.open(entry).convert("RGB").resize(self.resize_shape)
        img = np.array(img)

        ## Normalization idea, taken from here -> https://rwightman.github.io/pytorch-image-models/models/vision-transformer/ , mean = [0.5, 0.5, 0.5] and std = [0.5, 0.5, 0.5]
        
        if self.transform is not None:
          img = self.transform(img)
        
        ## Patches taken from here: https://discuss.pytorch.org/t/creating-3d-dataset-dataloader-with-patches/50861/2
        
        if  self.transform is None:
          img = ToTensor()(img)

        img = img.unfold(2,self.x_patch,self.y_patch).unfold(1,self.x_patch,self.y_patch).contiguous().view(-1, 3*self.x_patch*self.y_patch)
        return img, label