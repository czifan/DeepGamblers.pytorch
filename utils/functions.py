# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from utils.utils import accuracy

def train(config, epoch, model, train_loader, criterion, optimizer, logger, reward, weight):
    model.train()
    epoch_loss = []
    epoch_metric = {'top1': [], 'top5': []}
    num_batch = len(train_loader)
    for batch_ind, (inputs, targets, item) in enumerate(train_loader):
        inputs = inputs.float().to(config.device)
        targets = targets.long().to(config.device)
        outputs = model(inputs)
        if epoch >= config.pretrain:
            outputs = F.softmax(outputs, dim=1)
            outputs, reservation = outputs[:, :-1], outputs[:, -1]
            gain = torch.gather(outputs, dim=1, index=targets.unsqueeze(dim=1)).squeeze()
            doubling_rate = (gain.add(reservation.div(reward))).log()
            weight_item = np.zeros(inputs.shape[0])
            for ind, item_i in enumerate(item):
                weight_item[ind] = weight[item_i.item()]
            loss = 0.
            for ind, w in enumerate(weight_item):
                #print(weight_item[ind], (-doubling_rate[ind]))
                loss += (-doubling_rate[ind]) * weight_item[ind] / doubling_rate.shape[0]
            if config.hard_mining:
                temp = reservation / reservation.sum() - reservation.mean() / reservation.sum()
                #temp = (temp - temp.mean()) / temp.std() * 0.3
                for item_i, temp_i in zip(item, temp):
                    weight[item_i.item()] = 1 + temp_i.item()
        else:
            loss = criterion(outputs[:, :-1], targets)
        epoch_loss.append(loss.item())

        top1, top5 = accuracy(outputs.data, targets.data, topk=(1, 2))
        epoch_metric['top1'].append(top1.item())
        epoch_metric['top5'].append(top5.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info('[{:03d}] {:04d}/{:04d} Loss={:.4f}({:.4f}) Top1={:.2f}({:.2f}) Top5={:.2f}({:.2f})'.format(
            epoch, batch_ind, num_batch,
            epoch_loss[-1], np.mean(epoch_loss),
            epoch_metric['top1'][-1], np.mean(epoch_metric['top1']),
            epoch_metric['top5'][-1], np.mean(epoch_metric['top5'])))

    return epoch_loss, epoch_metric, weight
    
def test(config, epoch, model, test_loader, criterion, logger, reward):
    model.eval()
    epoch_loss = []
    epoch_metric = {'top1': [], 'top5': []}
    num_batch = len(test_loader)
    for batch_ind, (inputs, targets, _) in enumerate(test_loader):
        with torch.no_grad():
            inputs = inputs.float().to(config.device)
            targets = targets.long().to(config.device)
            outputs = model(inputs)
            if epoch >= config.pretrain:
                outputs = F.softmax(outputs, dim=1)
                outputs, reservation = outputs[:, :-1], outputs[:, -1]
                gain = torch.gather(outputs, dim=1, index=targets.unsqueeze(dim=1)).squeeze()
                doubling_rate = (gain.add(reservation.div(reward))).log()
                loss = -doubling_rate.mean()
            else:
                loss = criterion(outputs[:, :-1], targets)
            epoch_loss.append(loss.item())

            top1, top5 = accuracy(outputs.data, targets.data, topk=(1, 2))
            epoch_metric['top1'].append(top1.item())
            epoch_metric['top5'].append(top5.item())

        logger.info('[test {:03d}] {:04d}/{:04d} Loss={:.4f}({:.4f}) Top1={:.2f}({:.2f}) Top5={:.2f}({:.2f})'.format(
            epoch, batch_ind, num_batch,
            epoch_loss[-1], np.mean(epoch_loss),
            epoch_metric['top1'][-1], np.mean(epoch_metric['top1']),
            epoch_metric['top5'][-1], np.mean(epoch_metric['top5'])))

    return epoch_loss, epoch_metric


def evaluate(config, model, test_loader):
    model.eval()
    abortion_results = [[], []]
    for batch_ind, (inputs, targets, _) in enumerate(test_loader):
        inputs = inputs.float().to(config.device)
        targets = targets.long().to(config.device)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        outputs, reservation = outputs[:, :-1], outputs[:, -1]
        values, predictions = outputs.data.max(dim=1)
        abortion_results[0].extend(list(reservation.detach().cpu().numpy()))
        abortion_results[1].extend(list(predictions.eq(targets).detach().cpu().numpy()))
    def shuffle_list(list, seed=10):
        random.seed(seed)
        random.shuffle(list)
    shuffle_list(abortion_results[0])
    shuffle_list(abortion_results[1])
    abortion = torch.tensor(abortion_results[0])
    correct = torch.tensor(abortion_results[1])
    abortion_valid, abortion = abortion[:2000], abortion[2000:]
    correct_valid, correct = correct[:2000], correct[2000:]
    results_valid = []
    results = []
    bisection_method(config, abortion_valid, correct_valid, results_valid)
    bisection_method(config, abortion, correct, results)
    f = open(os.path.join(config.output_dir, 'coverage VS err.csv'), 'w')
    for idx, result in enumerate(results):
        f.write('valid{:.0f},{:.2f},{:.3f}\n'.format(config.coverages_list[idx],results_valid[idx][0]*100.,(1-results_valid[idx][1])*100))
        f.write('test{:.0f},{:.2f},{:.3f}\n'.format(config.coverages_list[idx],results[idx][0]*100.,(1-results[idx][1])*100))
    return

def bisection_method(config, abortion, correct, results):
    upper = 1.
    while True:
        mask_up = abortion <= upper
        passed_up = torch.sum(mask_up.long()).item()
        if passed_up / len(correct) * 100. < config.coverages_list[0]:
            upper *= 2.
        else:
            break
    test_thres = 1.
    for coverage in config.coverages_list:
        mask = abortion <= test_thres
        passed = torch.sum(mask.long()).item()
        lower = 0.
        while math.fabs(passed / len(correct) * 100. - coverage) > 0.3:
            if passed / len(correct) * 100. > coverage:
                upper = min(test_thres, upper)
                test_thres = (test_thres + lower) / 2
            elif passed / len(correct) * 100. < coverage:
                lower = max(test_thres, lower)
                test_thres = (test_thres + upper) / 2
            mask = abortion <= test_thres
            passed = torch.sum(mask.long()).item()
        masked_correct = correct[mask]
        correct_data = torch.sum(masked_correct.long()).item()
        passed_acc = correct_data / passed
        results.append((passed / len(correct), passed_acc))

def save_data(config):
    save=open('{}_{}.csv'.format(config.dataset, config.arch),'w')
    save.write('0,100val.,100test,99v,99t,98v,98t,97v,97t,95v,95t,90v,90t,85v,85t,80v,80t,75v,75t,70v,70t,60v,60t,50v,50t,40v,40t,30v,30t,20v,20t,10v,10t\n')
    for reward in config.rewards_list:
        f = open(os.path.join(config.output_dir, 'coverage VS err.csv') ,'r')
        content = f.read()
        lines = content.split('\n')
        save.write('o={:.2f},'.format(reward))
        for idx in range(len(config.coverages_list)):
            save.write('{},'.format(lines[2*idx].split(',')[2]))
            save.write('{},'.format(lines[2*idx+1].split(',')[2]))
        save.write('\n')
        f.close()


