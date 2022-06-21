from inference import inference
import linecache

ckpt_path = linecache.getline(r"/cluster/home/emanete/dental_imaging/checkpoints_and_scores/scores", 1).strip()
best_val_loss = linecache.getline(r"/cluster/home/emanete/dental_imaging/checkpoints_and_scores/scores", 2).strip()
inference(ckpt_path, best_val_loss, 64)
