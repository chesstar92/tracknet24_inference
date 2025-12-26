# tracknet24_inference
Inference for 2 public implementations of Tracknetv2 and Tracknetv4 (volleyball)


## how to run
(from top folder)
./setup.sh

**NB!** setup.sh doesn't cover download of dataset(s)

### both models, visualize, measure accuracy, dataset_orel

uv run python inference_unified.py \
  --model wasb \
  --dataset_orel \
  --test_input_path <dataset_root>/datasets/orel/test/4m2w_20250620_transmash_3_far/video/4m2w_20250620_transmash_3_far_005.mp4 \
  --output_path output \
  --visualize \
  --accuracy \
  --annotations_path <dataset_root>/datasets/orel/test/4m2w_20250620_transmash_3_far/csv/4m2w_20250620_transmash_3_far_005_predict_ball.csv

### both models, visualize, measure accuracy, dataset_wasb

uv run python inference_unified.py \
  --model wasb vballnet \
  --dataset_deep_activity_rec \
  --test_input_path <dataset_root>/datasets/wasb/test/videos/50/6070 \
  --output_path output \
  --visualize \
  --accuracy \
  --annotations_path <dataset_root>/datasets/wasb/test/volleyball_ball_annotation/50/6070.txt

### wasb model, visualize, dataset_orel (for custom video use 'dataset_orel')

uv run python inference_unified.py \
  --model vballnet \
  --dataset_orel \
  --test_input_path /my/custom/video.mp4 \
  --output_path output
