# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import pandas as pd

from torchvision import transforms

from hw_grapheme.predict_utils import bengali_predict
from hw_grapheme.model import EfficientNet_0

# +
test_data_paths = [
     '../data/train_image_data_0.parquet',
     '../data/train_image_data_2.parquet',
#     '../data/test_image_data_0.parquet',
#     '../data/test_image_data_1.parquet',
#     '../data/test_image_data_2.parquet',
#     '../data/test_image_data_3.parquet'
]

model_archs_weights = [
    (EfficientNet_0, "../model_weights/eff_0_with_mixup_cutmix/fold_0/eff_0_high_acc.pth"),
    (EfficientNet_0, "../model_weights/eff_0_with_mixup_cutmix/fold_2/eff_0_high_acc.pth"),
]

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # transforms.Normalize([0.0692], [0.2051])
])
# -

result = bengali_predict(
    test_data_paths, model_archs_weights, transforms,
    batch_size=128, n_workers=2
)

result

sub_df = pd.DataFrame({'row_id': result[0], 'target': result[1]})
#sub_df.to_csv('submission.csv', index=False)

sub_df


