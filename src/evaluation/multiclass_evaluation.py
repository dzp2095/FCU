# encoding: utf-8
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch 
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from src.utils.metric_logger import EMAMetricLogger
from src.evaluation.evaluation_strategy import EvaluationStrategy


import logging

def compute_metrics_multiclass(gt, pred_labels, pred_probs):
    roc_auc, acc, pre, recal, f1 = 0, 0, 0, 0, 0
    gt_np = np.asarray(gt)
    pred_labels_np = np.asarray(pred_labels)
    pred_probs_np = np.asarray(pred_probs)
    try:
        acc = accuracy_score(gt_np, pred_labels_np)
        res = classification_report(gt_np, pred_labels_np, output_dict=True)['macro avg']
        recal = res['recall']
        pre = res['precision']
        f1 = res['f1-score']
        roc_auc = roc_auc_score(gt_np, pred_probs_np, average='macro', multi_class='ovr')
    except ValueError as error:
        logging.exception(error)

    return roc_auc, acc, pre, recal, f1


class MulticlassEvaluationStrategy(EvaluationStrategy):
    def __init__(self, cfg):
        self.cfg = cfg
        
    def validate(self, model, data_loader, device, loss_fn) -> dict:
        return self.custom_eval(model, data_loader, device, loss_fn, "val")

    def test(self, model, data_loader, device, loss_fn) -> dict:
        return self.custom_eval(model, data_loader, device, loss_fn, "test")
    
    def custom_eval(self, model, data_loader, device, loss_fn, prefix) -> dict:
        roc_auc, acc, pre, recal, f1, loss = self._run(model, data_loader, device, loss_fn)
        metric_dict = {}
        metric_dict[f"{prefix}/auc_avg"] =  roc_auc
        metric_dict[f"{prefix}/f1_avg"] =  f1
        metric_dict[f"{prefix}/acc"] =  acc
        metric_dict[f"{prefix}/precision"] =  pre
        metric_dict[f"{prefix}/recall"] =  recal
        metric_dict[f"{prefix}/loss"] =  loss
        return metric_dict
    
    def unlearn_eval(self, model, test_data_loader, forgotten_dataloader, remembered_data_loader, device, loss_fn) -> dict:
        test_metrics = self.test(model, test_data_loader, device, loss_fn)
        test_metrics["test/error"] = 1 - test_metrics["test/acc"]
        forgotten_metrics = self.custom_eval(model, forgotten_dataloader, device, loss_fn, "forgotten")
        forgotten_metrics["forgotten/error"] = 1 - forgotten_metrics["forgotten/acc"]
        remembered_metrics = self.custom_eval(model, remembered_data_loader, device, loss_fn, "remembered")
        remembered_metrics["remembered/error"] = 1 - remembered_metrics["remembered/acc"]
        return {**test_metrics, **forgotten_metrics, **remembered_metrics}
        
    def _run(self, model, data_loader, device, loss_fn):
        model.eval()
        model.to(device=device)
        num_val_batches = len(data_loader)
        gt = []
        pred_labels = []
        pred_probs = []
        m_logger = EMAMetricLogger()
        with torch.no_grad():
            # iterate over the validation set
            for _, images, labels in tqdm(data_loader, total=num_val_batches, desc='Evaluation', unit='batch', leave=False):
                # move images and labels to correct device and type
                images = images.to(device=device)
                labels = labels.to(device=device)
                _, pred = model(images)
                loss = loss_fn(pred, labels)
                m_logger.update(loss=loss)
                output = F.one_hot(torch.argmax(pred, dim=1), num_classes = labels.shape[1])
                prob = F.softmax(pred, dim=1)
                gt = gt + labels.detach().cpu().numpy().tolist()
                pred_labels = pred_labels + output.detach().cpu().numpy().tolist()
                pred_probs = pred_probs + prob.detach().cpu().numpy().tolist()
        roc_auc, acc, pre, recal, f1 = compute_metrics_multiclass(gt, pred_labels,pred_probs)

        return roc_auc, acc, pre, recal, f1, m_logger.loss.global_avg

