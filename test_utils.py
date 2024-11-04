
from torch.nn import PairwiseDistance


def get_bn_statis(model, domain_id):
    means = []
    vars = []
    for name, param in model.state_dict().items():
        if 'bns.{}.running_mean'.format(domain_id) in name:
            means.append(param.clone())
        elif 'bns.{}.running_var'.format(domain_id) in name:
            vars.append(param.clone())
    return means, vars



def cal_distance(means_1, means_2, vars_1, vars_2):
    pdist = PairwiseDistance(p=2)
    dis = 0
    for (mean_1, mean_2, var_1, var_2) in zip(means_1, means_2, vars_1, vars_2):
        dis += (pdist(mean_1.reshape(1, mean_1.shape[0]), mean_2.reshape(1, mean_2.shape[0])) + pdist(var_1.reshape(1, var_1.shape[0]), var_2.reshape(1, var_2.shape[0])))
    return dis.item()