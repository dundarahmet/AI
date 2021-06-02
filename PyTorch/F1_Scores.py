import torch

def f1_scores_pytorch(y, y_hat, device, for_mini_batch=False):
    epsilon = 1e-8
    class_matrix = torch.zeros(y.shape[-1], 3, dtype=torch.float32).to(device) # false_positive, false_negative, true_positive

    false_negative = torch.sum(torch.bitwise_and((y == 1), (y_hat == 0)), dim=0)
    false_positive = torch.sum(torch.bitwise_and((y == 0), (y_hat == 1)), dim=0)
    true_positive = torch.sum(torch.bitwise_and((y == 1), (y_hat == 1)), dim=0)

    class_matrix[:, 0] += false_positive
    class_matrix[:, 1] += false_negative
    class_matrix[:, 2] += true_positive

    macro_precision = class_matrix[:, 2] / (torch.sum(class_matrix[:, [0, 2]], dim=1) + epsilon)
    macro_recall = class_matrix[:, 2] / (torch.sum(class_matrix[:, [1, 2]], dim=1) + epsilon)

    macro_f1_score = torch.sum((2 * macro_precision * macro_recall) / ((macro_precision + macro_recall) + epsilon)).item() / y.shape[-1]

    if for_mini_batch:
        return false_negative.sum().item(), false_positive.sum().item(), true_positive.sum().item(), macro_f1_score
    else:
        micro_precision = torch.sum(class_matrix[:, 2]) / (torch.sum(class_matrix[:, [0, 2]]) + epsilon)
        micro_recall = torch.sum(class_matrix[:, 2]) / (torch.sum(class_matrix[:, 1:]) + epsilon)
        micro_f1_score = ((2 * micro_precision * micro_recall) / (micro_precision + micro_recall)).item()
        return micro_f1_score, macro_f1_score
