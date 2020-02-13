# kaggle_bengali_recognition

model summary excel and model weights: https://oworkspace-my.sharepoint.com/personal/tonghu_oocl_com/_layouts/15/onedrive.aspx?viewid=a1352ee7-5566-48ae-8275-879ac9c7978d&id=%2Fpersonal%2Ftonghu_oocl_com%2FDocuments%2Fkaggle%2Fbengali

# Pre-process data
1. Run `data_processing_final.ipynb` (Run until the Step by Step cell)
2. Run for from `train_image_data_0.parquet` to `train_image_data_3.parquet`
3. Save the pre-processed pickle data into `data/processed_data/size_224/`

# Run training 

Suggest running `model_3_efficient_net.ipynb`

# Run inference

In `model_weights/eff_0_baseline` or `model_weights/eff_0_with_mixup_cutmix`, there are 5 folds of trained model weights.

Run `kaggle_submission.ipynb` for inference.
