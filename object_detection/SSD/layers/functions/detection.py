import torch
from torch._C import T
from torch.autograd import Function
from ..box_utils import decode, nms
from data import voc as cfg

class Detect(Function):

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classess = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        num = loc_data.size(0)  # batch size
        num_prior = prior_data.size(0)
        output = torch.zeros(num, self.num_classess, self.top_k, 5)
        conf_preds = conf_data.view(num, num_prior, self.num_classess).transpose(2,1)

        #decode predictions into bboxes
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_score = conf_preds[i].clone()

            for cl in range(1, self.num_classess):
                c_mask = conf_score[cl].gt(self.conf_thresh)
                scores = conf_score[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1,4)

                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]),1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


