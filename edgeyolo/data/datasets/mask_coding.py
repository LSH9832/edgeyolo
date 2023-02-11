import numpy as np
import torch


def encode_mask(segments: list, max_obj_num=120, max_edge_num=5, max_point_num=1000):
    pad_mask = - np.ones([max_obj_num, max_edge_num, max_point_num, 2])
    for i, obj in enumerate(segments):
        for j, edge in enumerate(obj):
            for k, point in enumerate(edge):
                if i < max_obj_num and j < max_edge_num and k < max_point_num:
                    pad_mask[i, j, k] = point
    return pad_mask


def decode_mask(segments: np.ndarray or torch.Tensor, data_type=float):
    segms = []
    for obj in segments:
        objs = []
        for edge in obj:
            if isinstance(edge, torch.Tensor):
                edge = edge.cpu().numpy()
            elif not isinstance(edge, np.ndarray):
                edge = np.array(edge)

            reserve_mask = (edge[..., 0] + edge[..., 1]) >= 0
            if len(reserve_mask):
                real_edge = edge[reserve_mask]
                if len(real_edge):
                    objs.append(real_edge.astype(data_type))
                # objs.append(np.vstack([edge[reserve_mask].astype(data_type), edge[reserve_mask][0:1].astype(data_type)]))
        if len(objs):
            segms.append(objs)
    return segms


if __name__ == '__main__':
    a = np.array([[[[0.0, 0.0],[0.1, 0.2],[0.3, 0.4], [0.5, 0.6], [-1, -1], [0.7, 0.8]],
                   [[-1, -1], [-1, -1], [0.3, 0.4], [0.8, 0.9]]]])
    print(decode_mask(a))
