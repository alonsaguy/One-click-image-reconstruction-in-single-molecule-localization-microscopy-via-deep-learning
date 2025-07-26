
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from DS3Dplus.training_utils import TorchTrainer
from datetime import datetime
from DS3Dplus.ds3d_utils import MyDataset, KDE_loss3D
from DS3Dplus.ds3d_utils import LON as Net
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time

np.random.seed(66)
torch.manual_seed(88)


if torch.cuda.device_count()>1:  # on server
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    td_folder = os.path.join(os.path.join(os.getcwd(), 'training_data'), 'laminB1_us2')

else:  # desktop
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    td_folder = os.path.join(os.path.join(os.getcwd(), 'training_data'), 'laminB1_us2')

path_save = os.path.join(os.getcwd(), 'training_results')
if not (os.path.isdir(path_save)):
    os.mkdir(path_save)



batch_size = 16
lr = 0.0005
params_train = {'batch_size': batch_size, 'shuffle': True}
params_validate = {'batch_size': batch_size, 'shuffle': True}

x_folder = os.path.join(td_folder, 'x')
x_list = os.listdir(x_folder)
num_x = len(x_list)
with open(os.path.join(td_folder, 'y.pickle'), 'rb') as handle:
    labels = pickle.load(handle)

partition = {'train': x_list[:int(num_x*0.9)], 'validate': x_list[int(num_x*0.9):]}
train_ds = MyDataset(x_folder, partition['train'], labels)
train_dl = DataLoader(train_ds, **params_train)
validate_ds = MyDataset(x_folder, partition['validate'], labels)
validate_dl = DataLoader(validate_ds, **params_validate)

D, us_factor, maxv = labels['volume_size'][0], labels['us_factor'], labels['blob_maxv']
model = Net(D=D, us_factor=us_factor, maxv=maxv).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'# of trainable parameters: {n_params}')


optimizer = Adam(list(model.parameters()), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True, min_lr=1e-6)  # verbose True
my_loss_func = torch.nn.MSELoss()
# my_loss_func = KDE_loss3D(sigma=0.5, device=device)
trainer = TorchTrainer(model, my_loss_func, optimizer, lr_scheduler=scheduler, device=device)

time_now = datetime.today().strftime('%m-%d_%H-%M')
net_file = 'net_'+time_now+'.pt'
checkpoints = dict(file_name=os.path.join(path_save, net_file),
                   net=Net(D=D, us_factor=us_factor, maxv=maxv),
                   state_dict=None,
                   note=' '
                   )

t0 = time.time()
fit_results = trainer.fit(train_dl, validate_dl, num_epochs=50, checkpoints=checkpoints, early_stopping=4)

fit_file = 'fit_'+time_now+'.pickle'
with open(os.path.join(path_save, fit_file), 'wb') as handle:
    pickle.dump(fit_results, handle)

t1 = time.time()

print(f'finished training in {t1-t0}s.')






