"""solver.py"""
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import os
from tqdm import tqdm

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

class Attack(object):
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion

    def fgsm(self, x, y, targeted=False, eps=0.03, x_val_min=0, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        h_adv = self.net(x_adv)
        
        cost = self.criterion(h_adv, y)
        if not targeted:
            cost = -cost

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.zero_()
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

        h = self.net(x)
        h_adv = self.net(x_adv)

        return x_adv, h_adv, h

    def i_fgsm(self, x, y, targeted=False, eps=0.03, alpha=1, iteration=1, x_val_min= -1, x_val_max=1):
        self.net.train()  # cần thiết nếu dùng LSTM + CuDNN

        print(f"[INPUT] x: {x.shape}, y: {y.shape}")
        x_adv = Variable(x.data.clone(), requires_grad=True)

        for i in range(iteration):
            out = self.net(x_adv)
            print(f"[{i}] net(x_adv) output: {type(out)}")

            # Xử lý output nếu là tuple (ví dụ LSTM)
            if isinstance(out, tuple):
                h_adv, _ = out
            else:
                h_adv = out

            print(f"[{i}] h_adv shape: {h_adv.shape}, y shape: {y.shape}")

            cost = self.criterion(h_adv, y)
            print(f"[{i}] loss: {cost.item():.4f}")

            if not targeted:
                cost = -cost

            self.net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.zero_()

            cost.backward()

            step = alpha * x_adv.grad.sign()
            x_adv = x_adv + step if targeted else x_adv - step
            x_adv = where(x_adv > x + eps, x + eps, x_adv)
            x_adv = where(x_adv < x - eps, x - eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data.clone(), requires_grad=True)

            print(f"[{i}] x_adv shape: {x_adv.shape}")

        h = self.net(x)
        h_adv = self.net(x_adv)

        return x_adv, h_adv, h
    


class Solver(object):
    def __init__(self, args):
        self.args = args

        self.cuda = (args.cuda and torch.cuda.is_available())
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.eps = args.eps
        self.lr = args.lr
        self.y_dim = args.y_dim
        self.dataset = args.dataset
        self.data_loader = args.data_loader

        self.model = args.model.to(self.device)
        self.net = self.model  # For compatibility

        self.attack = Attack(self.net, criterion=args.criterion)
        self.optimizer = args.optimizer
        self.criterion = args.criterion

        self.save_path = args.save_path
        self.ckpt_dir = Path(args.save_path) / "checkpoints"
        self.summary_dir = Path(args.save_path) / "summary"
        self.output_dir = Path(args.save_path) / "output"
        self.global_iter = 0
        self.global_epoch = 0
        self.history = {}
        self.visdom = False

    def train(self, train_loader, val_loader, num_epochs=10):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        for epoch in range(num_epochs):
            running_loss = 0.0
            progress_bar = tqdm(train_loader, 
                                desc=f'Epoch {epoch+1}/{num_epochs}', 
                                unit='batch')
            for batch_id, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix({'loss': running_loss/(batch_id+1)})

            print(f'Epoch {epoch+1} - Avg Loss: {running_loss/len(train_loader):.4f}')

        # Validation
        self.model.eval()
        correct_pred = 0
        total_pred = 0
        with torch.no_grad():
            test_bar = tqdm(val_loader, desc='Evaluating', unit='batch')
            for inputs, labels in test_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs  = self.model(inputs)
                _, pred = torch.max(outputs.data, 1)
                total_pred += labels.size(0)
                correct_pred += (pred == labels).sum().item()
                test_bar.set_postfix({'acc': f'{100*correct_pred/total_pred:.2f}%'})

        print("Saving model")
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f'{self.model.model_name}.pth'))

    def test(self, testloader):
        self.model.eval()
        correct_pred = 0
        total_pred = 0

        with torch.no_grad():
            test_bar = tqdm(testloader, desc='Testing', unit='batch')
            for inputs, labels in test_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, pred = torch.max(outputs.data, 1)
                total_pred += labels.size(0)
                correct_pred += (pred == labels).sum().item()
                test_bar.set_postfix({'acc': f'{100*correct_pred/total_pred:.2f}%'})

    def sample_data(self, test_loader, num_sample=100):
        x_true = []
        y_true = []
        total = 0
        for x, y in test_loader:
            x_true.append(x)
            y_true.append(y)
            total += x.shape[0]
            if total >= num_sample:
                break

        x_true = torch.cat(x_true, dim=0)[:num_sample].to(self.device)
        y_true = torch.cat(y_true, dim=0)[:num_sample].to(self.device)
        return x_true, y_true
        

    def FGSM(self, x, y_true, y_target=None, eps=0.03, alpha=2/255, iteration=1):
        self.set_mode('train')

        x = Variable(x, requires_grad=True)
        y_true = Variable(y_true, requires_grad=False)
        targeted = y_target is not None
        if targeted:
            y_target = Variable(y_target, requires_grad=False)

        h = self.net(x)
        prediction = h.max(1)[1]
        accuracy = torch.eq(prediction, y_true).float().mean()
        cost = F.cross_entropy(h, y_true)

        if iteration == 1:
            x_adv, h_adv, h = self.attack.fgsm(x, y_target if targeted else y_true, targeted, eps)
        else:
            x_adv, h_adv, h = self.attack.i_fgsm(x, y_target if targeted else y_true, targeted, eps, alpha, iteration)

        prediction_adv = h_adv.argmax(dim=1)
        accuracy_adv = torch.eq(prediction_adv, y_true).float().mean()
        cost_adv = F.cross_entropy(h_adv, y_true)

        changed = (prediction != prediction_adv)
        print(f"Changed Predictions Indexes: {torch.nonzero(changed).squeeze().tolist()}")

        self.set_mode('train')

        return x_adv.data, (accuracy.item(), cost.item(), accuracy_adv.item(), cost_adv.item())

    def generate(self, num_sample=100, target=-1, epsilon=0.03, alpha=2/255, iteration=1):
        self.set_mode('eval')
        x_true, y_true = self.sample_data(self.data_loader['test'], num_sample)

        print(f"x_true shape: {x_true.shape}, y_true shape: {y_true.shape}")

        y_target = None
        if isinstance(target, int) and (target in range(self.y_dim)):
            y_target = torch.LongTensor(y_true.size()).fill_(target).to(self.device)

        print(f"Generate attack on batch size: {x_true.shape[0]}")
        x_adv, values = self.FGSM(x_true, y_true, y_target, epsilon, alpha, iteration)
        accuracy, cost, accuracy_adv, cost_adv = values

        print('[BEFORE] accuracy : {:.2f} cost : {:.3f}'.format(accuracy, cost))
        print('[AFTER]  accuracy : {:.2f} cost : {:.3f}'.format(accuracy_adv, cost_adv))

        self.set_mode('train')

    def save_checkpoint(self, filename='ckpt.tar'):
        model_states = {'net': self.net.state_dict()}
        optim_states = {'optim': self.optimizer.state_dict()}
        states = {
            'iter': self.global_iter,
            'epoch': self.global_epoch,
            'history': self.history,
            'args': self.args,
            'model_states': model_states,
            'optim_states': optim_states,
        }
        file_path = self.ckpt_dir / filename
        torch.save(states, file_path.open('wb'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='best_acc.tar'):
        file_path = self.ckpt_dir / filename
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optimizer.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.net.train()
        elif mode == 'eval':
            self.net.eval()
        else:
            raise ValueError('mode error. It should be either train or eval')

# abvl