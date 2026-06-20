from __future__ import print_function, absolute_import
import time
import os.path as osp

from torch.nn import functional as F
import torch
import torch.nn as nn
from src.utils.reid_utils.meters import AverageMeter
from src.utils.reid_utils.feature_tools import *

from src.utils.reid_utils.make_loss import make_loss
from src.knowledge.loss.triplet_loss_transreid import TripletLoss
import copy

from src.knowledge.metric_learning.distance import cosine_similarity
class Trainer(object):
    def __init__(self,cfg,args, model, num_classes, writer=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        self.writer = writer
        self.AF_weight = args.AF_weight

        self.loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
        self.place_train_positive_tolerance = int(getattr(args, 'place_train_positive_tolerance', 0) or 0)
        self.place_ce_weight = float(getattr(args, 'place_ce_weight', 1.0) or 1.0)
        self.place_triplet_weight = float(getattr(args, 'place_triplet_weight', 1.0) or 1.0)
        self.vprtempo_raw_triplet_weight = float(getattr(args, 'vprtempo_raw_triplet_weight', 0.0) or 0.0)
        self.vprtempo_raw_infonce_weight = float(getattr(args, 'vprtempo_raw_infonce_weight', 0.0) or 0.0)
        self.vprtempo_raw_infonce_temp = float(getattr(args, 'vprtempo_raw_infonce_temp', 0.07) or 0.07)
        self.place_raw_distill_weight = float(getattr(args, 'place_raw_distill_weight', 0.0) or 0.0)
        self.place_raw_distill_freq = max(1, int(getattr(args, 'place_raw_distill_freq', 1) or 1))
        self.place_projected_distill_weight = float(getattr(args, 'place_projected_distill_weight', 0.0) or 0.0)
        self.place_projected_distill_freq = max(1, int(getattr(args, 'place_projected_distill_freq', 1) or 1))
        self.raw_triplet = TripletLoss(cfg.SOLVER.MARGIN)

      
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        self.vprtempo_distill_weight = getattr(args, 'vprtempo_distill_weight', 0.0)
        self.external_prototype_bank = None
        self.external_prototype_mask = None
        self.teacher_feature_bank = None
        self.prototype_add_num = 0
        amp_enabled = bool(getattr(args, 'amp', False)) and torch.cuda.is_available()
        self.amp_enabled = amp_enabled
        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
            self._autocast = torch.amp.autocast
            self._autocast_device = 'cuda'
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            self._autocast = torch.cuda.amp.autocast
            self._autocast_device = None

    def set_external_prototypes(self, prototype_bank, add_num=0):
        self.prototype_add_num = add_num
        if prototype_bank is None or self.vprtempo_distill_weight <= 0:
            self.external_prototype_bank = None
            self.external_prototype_mask = None
            return
        self.external_prototype_bank = prototype_bank['bank'].cuda()
        self.external_prototype_mask = prototype_bank['mask'].cuda()

    def compute_prototype_distill_loss(self, features, targets):
        if self.external_prototype_bank is None or self.external_prototype_mask is None:
            return None
        local_targets = targets - self.prototype_add_num
        valid = (local_targets >= 0) & (local_targets < self.external_prototype_bank.size(0))
        if valid.sum().item() == 0:
            return None
        local_targets = local_targets[valid]
        sample_mask = self.external_prototype_mask[local_targets] > 0
        if sample_mask.sum().item() == 0:
            return None
        local_targets = local_targets[sample_mask]
        current_features = features[valid][sample_mask]
        if hasattr(self.model.module, 'distill_projector'):
            current_features = self.model.module.distill_projector(current_features)
        target_features = self.external_prototype_bank[local_targets]
        current_features = F.normalize(current_features, dim=1)
        target_features = F.normalize(target_features, dim=1)
        return 1.0 - (current_features * target_features).sum(dim=1).mean()

    def set_teacher_feature_bank(self, teacher_feature_bank):
        if teacher_feature_bank is None or float(getattr(self.args, 'place_teacher_sim_weight', 0.0) or 0.0) <= 0:
            self.teacher_feature_bank = None
            return
        self.teacher_feature_bank = {
            osp.normcase(osp.normpath(str(path))): F.normalize(feature.float(), dim=0).cpu()
            for path, feature in teacher_feature_bank.items()
        }

    def compute_teacher_similarity_loss(self, student_features, fnames):
        if self.teacher_feature_bank is None:
            return None
        teacher_features = []
        valid_indices = []
        for index, fname in enumerate(fnames):
            key = osp.normcase(osp.normpath(str(fname)))
            feature = self.teacher_feature_bank.get(key)
            if feature is not None:
                teacher_features.append(feature)
                valid_indices.append(index)
        min_batch = int(getattr(self.args, 'place_teacher_sim_min_batch', 4) or 4)
        if len(teacher_features) < min_batch:
            return None
        indices = torch.as_tensor(valid_indices, dtype=torch.long, device=student_features.device)
        student = F.normalize(student_features.index_select(0, indices).float(), dim=1)
        teacher = torch.stack(teacher_features, dim=0).to(student.device, non_blocking=True)
        teacher = F.normalize(teacher.float(), dim=1)
        temperature = max(float(getattr(self.args, 'place_teacher_sim_temp', 0.07) or 0.07), 1e-6)
        student_logits = torch.matmul(student, student.t()) / temperature
        teacher_logits = torch.matmul(teacher, teacher.t()) / temperature
        eye = torch.eye(student_logits.size(0), dtype=torch.bool, device=student_logits.device)
        student_logits = student_logits.masked_fill(eye, -1e4)
        teacher_logits = teacher_logits.masked_fill(eye, -1e4)
        teacher_prob = F.softmax(teacher_logits, dim=1)
        student_log_prob = F.log_softmax(student_logits, dim=1)
        return F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')

    def _extract_projected_feature(self, model, inputs):
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(outputs, list):
            outputs = outputs[0]
        return outputs

    def train(self, epoch, data_loader_train,  optimizer, training_phase,
              train_iters=200, add_num=0, old_model=None,         
              raw_memory_loader=None,
              ):

        self.model.train()
        # freeze the bn layer totally
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_raw_tr = AverageMeter()
        losses_raw_nce = AverageMeter()
        losses_raw_distill = AverageMeter()
        losses_projected_distill = AverageMeter()
        losses_teacher_sim = AverageMeter()
        tb_log_freq = int(getattr(self.args, 'tb_log_freq', 0) or 0)
        if tb_log_freq <= 0:
            tb_log_freq = max(1, int(getattr(self.args, 'print_freq', 200)))

        end = time.time()
        print_freq = max(1, int(getattr(self.args, 'print_freq', 200)))

        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            s_inputs, fnames, targets, cids, domains, = self._parse_data(train_inputs)
            targets += add_num
            autocast_kwargs = {'enabled': self.amp_enabled}
            if self._autocast_device is not None:
                autocast_kwargs['device_type'] = self._autocast_device
            with self._autocast(**autocast_kwargs):
                raw_feat = None
                raw_triplet_loss = None
                raw_infonce_loss = None
                teacher_sim_loss = None
                if (getattr(self.args, 'task_type', None) == 'place' and
                        getattr(self.args, 'MODEL', None) == 'vprtempo_snn' and
                        self.vprtempo_raw_triplet_weight > 0 and
                        hasattr(self.model.module, 'forward_with_raw')):
                    s_features, bn_feat, cls_outputs, feat_final_layer, raw_feat = self.model.module.forward_with_raw(s_inputs)
                else:
                    s_features, bn_feat, cls_outputs, feat_final_layer = self.model(s_inputs)

                '''calculate the base loss'''
                triplet_positive_mask = self._build_triplet_positive_mask(targets)
                loss_ce, loss_tp = self.loss_fn(
                    cls_outputs,
                    s_features,
                    targets,
                    target_cam=None,
                    triplet_positive_mask=triplet_positive_mask,
                )
                if getattr(self.args, 'task_type', None) == 'place':
                    loss = loss_ce * self.place_ce_weight + loss_tp * self.place_triplet_weight
                    if raw_feat is not None:
                        raw_triplet_loss = self.raw_triplet(
                            raw_feat,
                            targets,
                            normalize_feature=True,
                            positive_mask=triplet_positive_mask,
                        )[0]
                        loss = loss + raw_triplet_loss * self.vprtempo_raw_triplet_weight
                    else:
                        raw_triplet_loss = None
                    if raw_feat is not None and self.vprtempo_raw_infonce_weight > 0:
                        raw_infonce_loss = self._raw_supervised_contrastive_loss(
                            raw_feat,
                            triplet_positive_mask,
                            targets,
                        )
                        if raw_infonce_loss is not None:
                            loss = loss + raw_infonce_loss * self.vprtempo_raw_infonce_weight
                else:
                    loss = loss_ce + loss_tp
                distill_loss = self.compute_prototype_distill_loss(s_features, targets)
                if distill_loss is not None:
                    loss = loss + distill_loss * self.vprtempo_distill_weight
                teacher_sim_loss = self.compute_teacher_similarity_loss(raw_feat if raw_feat is not None else s_features, fnames)
                if teacher_sim_loss is not None:
                    loss = loss + teacher_sim_loss * float(getattr(self.args, 'place_teacher_sim_weight', 0.0) or 0.0)

                if old_model is not None:
                    with torch.no_grad():
                        s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = old_model(s_inputs, get_all_feat=True)
                    if isinstance(s_features_old, tuple):
                        s_features_old=s_features_old[0]
                    Affinity_matrix_new = self.get_normal_affinity(s_features)
                    Affinity_matrix_old = self.get_normal_affinity(s_features_old)
                    divergence = self.cal_KL(Affinity_matrix_new, Affinity_matrix_old, targets)
                    loss = loss + divergence * self.AF_weight

                raw_distill_loss = None
                projected_distill_loss = None
                if (raw_memory_loader is not None and old_model is not None and
                        self.place_raw_distill_weight > 0 and
                        ((i + 1) % self.place_raw_distill_freq == 0) and
                        hasattr(self.model.module, 'extract_raw_feature')):
                    memory_inputs, _, _, _ = self._parse_memory_data(raw_memory_loader.next())
                    with torch.no_grad():
                        teacher_raw = old_model.module.extract_raw_feature(memory_inputs)
                        teacher_raw = F.normalize(teacher_raw.float(), dim=1)
                    student_raw = self.model.module.extract_raw_feature(memory_inputs)
                    student_raw = F.normalize(student_raw.float(), dim=1)
                    raw_distill_loss = 1.0 - (student_raw * teacher_raw).sum(dim=1).mean()
                    loss = loss + raw_distill_loss * self.place_raw_distill_weight
                if (raw_memory_loader is not None and old_model is not None and
                        self.place_projected_distill_weight > 0 and
                        ((i + 1) % self.place_projected_distill_freq == 0)):
                    memory_inputs, _, _, _ = self._parse_memory_data(raw_memory_loader.next())
                    with torch.no_grad():
                        teacher_projected = self._extract_projected_feature(old_model, memory_inputs)
                        teacher_projected = F.normalize(teacher_projected.float(), dim=1)
                    student_projected = self._extract_projected_feature(self.model, memory_inputs)
                    student_projected = F.normalize(student_projected.float(), dim=1)
                    projected_distill_loss = 1.0 - (student_projected * teacher_projected).sum(dim=1).mean()
                    loss = loss + projected_distill_loss * self.place_projected_distill_weight

            optimizer.zero_grad(set_to_none=True)
            if self.amp_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            should_log = ((i + 1) % print_freq == 0) or ((i + 1) == train_iters)
            should_tb = self.writer is not None and ((((i + 1) % tb_log_freq) == 0) or ((i + 1) == train_iters))
            if should_log or should_tb:
                loss_ce_value = float(loss_ce.detach())
                loss_tp_value = float(loss_tp.detach())
                losses_ce.update(loss_ce_value)
                losses_tr.update(loss_tp_value)
                if raw_triplet_loss is not None:
                    losses_raw_tr.update(float(raw_triplet_loss.detach()))
                if raw_infonce_loss is not None:
                    losses_raw_nce.update(float(raw_infonce_loss.detach()))
                if raw_distill_loss is not None:
                    losses_raw_distill.update(float(raw_distill_loss.detach()))
                if projected_distill_loss is not None:
                    losses_projected_distill.update(float(projected_distill_loss.detach()))
                if teacher_sim_loss is not None:
                    losses_teacher_sim.update(float(teacher_sim_loss.detach()))
            if should_tb:
                global_step = epoch * train_iters + i
                self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=loss_ce_value,
                          global_step=global_step)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=loss_tp_value,
                          global_step=global_step)
                if raw_triplet_loss is not None:
                    self.writer.add_scalar(tag="loss/Loss_raw_tr_{}".format(training_phase),
                          scalar_value=float(raw_triplet_loss.detach()), global_step=global_step)
                if raw_infonce_loss is not None:
                    self.writer.add_scalar(tag="loss/Loss_raw_nce_{}".format(training_phase),
                          scalar_value=float(raw_infonce_loss.detach()), global_step=global_step)
                if raw_distill_loss is not None:
                    self.writer.add_scalar(tag="loss/Loss_raw_distill_{}".format(training_phase),
                          scalar_value=float(raw_distill_loss.detach()), global_step=global_step)
                if projected_distill_loss is not None:
                    self.writer.add_scalar(tag="loss/Loss_projected_distill_{}".format(training_phase),
                          scalar_value=float(projected_distill_loss.detach()), global_step=global_step)
                if teacher_sim_loss is not None:
                    self.writer.add_scalar(tag="loss/Loss_teacher_sim_{}".format(training_phase),
                          scalar_value=float(teacher_sim_loss.detach()), global_step=global_step)
                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                          global_step=global_step)
            if should_log:
                log_msg = ('Epoch: [{}][{}/{}]\t'
                           'Time {:.3f} ({:.3f})\t'
                           'Loss_ce {:.3f} ({:.3f})\t'
                           'Loss_tp {:.3f} ({:.3f})\t').format(
                               epoch, i + 1, train_iters,
                               batch_time.val, batch_time.avg,
                               losses_ce.val, losses_ce.avg,
                               losses_tr.val, losses_tr.avg,
                           )
                if raw_triplet_loss is not None:
                    log_msg += 'Loss_raw_tp {:.3f} ({:.3f})\t'.format(
                        losses_raw_tr.val,
                        losses_raw_tr.avg,
                    )
                if raw_infonce_loss is not None:
                    log_msg += 'Loss_raw_nce {:.3f} ({:.3f})\t'.format(
                        losses_raw_nce.val,
                        losses_raw_nce.avg,
                    )
                if raw_distill_loss is not None:
                    log_msg += 'Loss_raw_distill {:.3f} ({:.3f})\t'.format(
                        losses_raw_distill.val,
                        losses_raw_distill.avg,
                    )
                if projected_distill_loss is not None:
                    log_msg += 'Loss_projected_distill {:.3f} ({:.3f})\t'.format(
                        losses_projected_distill.val,
                        losses_projected_distill.avg,
                    )
                if teacher_sim_loss is not None:
                    log_msg += 'Loss_teacher_sim {:.3f} ({:.3f})\t'.format(
                        losses_teacher_sim.val,
                        losses_teacher_sim.avg,
                    )
                print(log_msg)

    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix
    def _parse_data(self, inputs):
        imgs, fnames, pids, cids, domains = inputs
        inputs = imgs.cuda(non_blocking=True)
        targets = pids.cuda(non_blocking=True)
        return inputs, fnames, targets, cids, domains
    def _parse_memory_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda(non_blocking=True)
        targets = pids.cuda(non_blocking=True)
        return inputs, targets, cids, domains
    def _raw_supervised_contrastive_loss(self, raw_feat, positive_mask, targets):
        features = F.normalize(raw_feat.float(), dim=1)
        logits = torch.matmul(features, features.t()) / max(self.vprtempo_raw_infonce_temp, 1e-6)
        batch_size = logits.size(0)
        eye = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
        if positive_mask is None:
            positives = targets.reshape(-1, 1).eq(targets.reshape(1, -1))
        else:
            positives = positive_mask.bool()
        positives = positives & (~eye)
        valid = positives.sum(dim=1) > 0
        if valid.sum().item() == 0:
            return None
        logits = logits.masked_fill(eye, -1e4)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        mean_log_prob_pos = (log_prob * positives.float()).sum(dim=1) / positives.sum(dim=1).clamp_min(1).float()
        return -mean_log_prob_pos[valid].mean()
    def _build_triplet_positive_mask(self, targets):
        if self.place_train_positive_tolerance <= 0 or getattr(self.args, 'task_type', None) != 'place':
            return None
        diff = torch.abs(targets.reshape(-1, 1) - targets.reshape(1, -1))
        positive_mask = diff <= int(self.place_train_positive_tolerance)
        positive_mask.fill_diagonal_(True)
        return positive_mask
    def cal_KL(self,Affinity_matrix_new, Affinity_matrix_old,targets):
        Gts = (targets.reshape(-1, 1) - targets.reshape(1, -1)) == 0  # Gt-matrix
        Gts = Gts.float().to(targets.device)
        '''obtain TP,FP,TN,FN'''
        attri_new = self.get_attri(Gts, Affinity_matrix_new, margin=0)
        attri_old = self.get_attri(Gts, Affinity_matrix_old, margin=0)

        '''# prediction is correct on old model'''
        Old_Keep = attri_old['TN'] + attri_old['TP']
        Target_1 = Affinity_matrix_old * Old_Keep
        '''# prediction is false on old model but correct on mew model'''
        New_keep = (attri_new['TN'] + attri_new['TP']) * (attri_old['FN'] + attri_old['FP'])
        Target_2 = Affinity_matrix_new * New_keep
        '''# both missed correct person'''
        Hard_pos = attri_new['FN'] * attri_old['FN']
        Thres_P = torch.maximum(attri_new['Thres_P'], attri_old['Thres_P'])
        Target_3 = Hard_pos * Thres_P

        '''# both false wrong person'''
        Hard_neg = attri_new['FP'] * attri_old['FP']
        Thres_N = torch.minimum(attri_new['Thres_N'], attri_old['Thres_N'])
        Target_4 = Hard_neg * Thres_N

        Target__ = Target_1 + Target_2 + Target_3 + Target_4
        Target = Target__ / (Target__.sum(1, keepdim=True))  # score normalization


        Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
        divergence=self.KLDivLoss(Affinity_matrix_new_log, Target)

        return divergence

    def get_attri(self, Gts, pre_affinity_matrix,margin=0):
        Thres_P=((1-Gts)*pre_affinity_matrix).max(dim=1,keepdim=True)[0]
        T_scores=pre_affinity_matrix*Gts

        TP=((T_scores-Thres_P)>margin).float()
        TP=torch.maximum(TP, torch.eye(TP.size(0)).to(TP.device))

        FN=Gts-TP

        Mapped_affinity=(1-Gts) +pre_affinity_matrix
        Mapped_affinity = Mapped_affinity+torch.eye(Mapped_affinity.size(0)).to(Mapped_affinity.device)
        Thres_N = Mapped_affinity.min(dim=1, keepdim=True)[0]
        N_scores=pre_affinity_matrix*(1-Gts)

        FP=(N_scores>Thres_N ).float()
        TN=(1-Gts) -FP
        attris={
            'TP':TP,
            'FN':FN,
            'FP':FP,
            'TN':TN,
            "Thres_P":Thres_P,
            "Thres_N":Thres_N
        }
        return attris

