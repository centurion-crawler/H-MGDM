import torch
from torch_scatter import scatter_add


def get_distance(pos, edge_index,smooth=1e-6):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)+smooth

def detect_nan(x):
    return torch.isnan(x).any()

def detect_inf(x):
    return torch.isinf(x).any()

# scatter_add dim=0 self[index[i][j]][j] += src[i][j]
def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.size(0)
    # print('edge_len:',edge_length.min(),edge_length.mean(),edge_length.max())
    # print('detect nan:',detect_inf(1. / edge_length),detect_nan(pos[edge_index[0]] - pos[edge_index[1]]))
    dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])  # (E, 3)
    # print('detect nan:',detect_nan(dd_dr))
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) #\
        # + scatter_add(- dd_dr * score_d, edge_index[1], dim=0, dim_size=N) # (N, 3)
    # score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N)
    return score_pos


def convert_cluster_score_d(cluster_score_d, cluster_pos, cluster_edge_index, cluster_edge_length, subgraph_index):
    """
    Args:
        cluster_score_d:    (E_c, 1)
        subgraph_index:     (N, )
    """
    cluster_score_pos = eq_transform(cluster_score_d, cluster_pos, cluster_edge_index, cluster_edge_length)  # (C, 3)
    score_pos = cluster_score_pos[subgraph_index]
    return score_pos


def get_angle(pos, angle_index):
    """
    Args:
        pos:  (N, 3)
        angle_index:  (3, A), left-center-right.
    """
    n1, ctr, n2 = angle_index  # (A, )
    v1 = pos[n1] - pos[ctr]  # (A, 3)
    v2 = pos[n2] - pos[ctr]
    inner_prod = torch.sum(v1 * v2, dim=-1, keepdim=True)  # (A, 1)
    length_prod = torch.norm(v1, dim=-1, keepdim=True) * torch.norm(v2, dim=-1, keepdim=True)  # (A, 1)
    angle = torch.acos(inner_prod / length_prod)  # (A, 1)
    return angle


def get_dihedral(pos, dihedral_index):
    """
    Args:
        pos:  (N, 3)
        dihedral:  (4, A)
    """
    n1, ctr1, ctr2, n2 = dihedral_index  # (A, )
    v_ctr = pos[ctr2] - pos[ctr1]  # (A, 3)
    v1 = pos[n1] - pos[ctr1]
    v2 = pos[n2] - pos[ctr2]
    n1 = torch.cross(v_ctr, v1, dim=-1)  # Normal vectors of the two planes
    n2 = torch.cross(v_ctr, v2, dim=-1)
    inner_prod = torch.sum(n1 * n2, dim=1, keepdim=True)  # (A, 1)
    length_prod = torch.norm(n1, dim=-1, keepdim=True) * torch.norm(n2, dim=-1, keepdim=True)
    dihedral = torch.acos(inner_prod / length_prod)
    return dihedral
