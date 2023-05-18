import torch
import torch.optim as optim
from tqdm import tqdm

from ..simsiam.simsiam import D
from ..optimizers import *

class Client:
    def __init__(self, client_id, model, dataloader, local_epochs, num_rounds):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.local_epochs = local_epochs
        self.total_epochs = local_epochs * num_rounds
        self.lr_scheduler = None

    def init_lrscheduler(self):
        # fixed parameters
        warmup_epochs = 10
        warmup_lr = 0
        base_lr = 0.03
        final_lr = 0
        momentum = 0.9
        weight_decay = 0.0005
        batch_size = 64

        optimizer = get_optimizer(
            'sgd', self.model, 
            lr=base_lr*batch_size/256, 
            momentum=momentum,
            weight_decay=weight_decay)

        lr_scheduler = LR_Scheduler(
            optimizer, warmup_epochs, warmup_lr*batch_size/256, 
            self.total_epochs, base_lr*batch_size/256, final_lr*batch_size/256, 
            len(self.dataloader),
            constant_predictor_lr=True
        )
        return optimizer, lr_scheduler


    def client_update(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.model.train()
        
        # optimizer = optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005)

        #if self.lr_scheduler is None:
        self.optimizer, self.lr_scheduler = self.init_lrscheduler()

        global_progress = tqdm(range(0, self.local_epochs), desc=f'Training client {self.client_id + 1}')
        for epoch in global_progress:
            self.model.train()
            
            local_progress=tqdm(self.dataloader, desc=f'Epoch {epoch + 1}/{self.local_epochs}')
            for idx, data in enumerate(local_progress):
                images = data[0]

                image = images[0][0]
                views = images[1]

                self.optimizer.zero_grad()
                data_dict = self.model.forward(image.to(self.device, non_blocking=True), views)
                loss = data_dict['loss'].mean()

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                
                local_progress.set_postfix(data_dict)

            epoch_dict = {"epoch":epoch}
            global_progress.set_postfix(epoch_dict) 
            
        # save learning rate so it can be used in next round
        print('last learning rate: ', self.optimizer.param_groups[0]['lr'])