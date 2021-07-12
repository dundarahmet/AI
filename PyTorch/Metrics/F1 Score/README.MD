# For f1_scores_pytorch(y, y_hat, device, for_mini_batch=False) function in F1_Scores.py:
- If you just want to use micro and macro averaged f1 scores without mini batch, for_mini_batch argument must be False. 
- But, if you want to use this function in training with mini batch, use the function following:
  
```
  def train_one_epoch(...):
    batch_count = 0 // must be created to use the function
    epoch_fn = 0    // must be created to use the function
    epoch_fp 0      // must be created to use the function
    epoch_tp 0       // must be created to use the function
    epoch_macro_f1_scores = 0 // must be created to use the function
    for images, labels in train_loader:
          ...
          output = model(...)
          with torch.no_grad():
            # for_mini_batch must be true
            fn, fp, tp, macro = f1_scores_pytorch(labels, outputs, device, for_mini_batch = True)
            epoch_fn += fn
            epoch_fp += fp
            epoch_tp += tp
            epoch_macro_f1_scores += macro
    
    # after first epoch
    micro_recall = (epoch_tp) / (epoch_fn + epoch_tp + 1e-8)    # 1e-8 is using to avoid zero division
    micro_precision = (epoch_tp) / (epoch_fp + epoch_tp + 1e-8)
    micro_f1_scores = (2 * micro_recall * micro_precision) / (micro_recall + micro_presicion + 1e-8)
    epoch_macro_f1_scores = epoch_macro_f1_scores / batch_count
    return epoch_macro_f1_scores, micro_f1_scores, ...
```
