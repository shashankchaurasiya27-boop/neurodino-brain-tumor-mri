Run Live Inference in Terminal
Run live inference directly against the certified model checkpoint on Apple Silicon / CPU:

bash
python3 predictions_showcase_50/run_live_inference_demo.py --limit 10
To run prediction on a single specific image:

bash
python3 predictions_showcase_50/run_live_inference_demo.py --image "predictions_showcase_50/raw_mri_images/glioma/Te-gl_163.jpg"
