# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

class Interpret():
    def __init__(self, model):
        self.model = model
#         model = models.resnet18(pretrained=True)
        model = model.eval()
    
    def __call__(self, image):
        """
        images: 3D image (Channel, Height , Width)
        """
        input = image.unsqueeze(0)
        output = model(input)
        output = tuple(map(lambda x: F.softmax(x), (output)))
        pred=tuple(map(lambda x: torch.topk(x,1), (b)))
        score1, idx1 =pred[0]
        score2, idx2 =pred[1]
        score3, idx3 =pred[2]
        prediction_score = (score1, score2, score3)
        pred_label_idx = (idx1, idx2, idx3)
        prediction_score =  tuple(map(lambda x:x.item(), prediction_score))
        pred_label_idx = tuple(map(lambda x:x.item(), pred_label_idx))
#         prediction_score, pred_label_idx = torch.topk(output, 3)
        print(f'Top Prediction Score: {prediction_score}, Label Index: {pred_label_idx}')
        pred_label_idx = np.array(pred_label_idx[0])
#         _, pred_label_idx = torch.topk(output, 1)
        torch.manual_seed(0)
        np.random.seed(0)

        gradient_shap = GradientShap(model)

        # Defining baseline distribution of images
        rand_img_dist = torch.cat([input * 0, input * 1])

        attributions_gs = gradient_shap.attribute(input,
                                                  n_samples=50,
                                                  stdevs=0.0001,
                                                  baselines=rand_img_dist,
                                                  target=pred_label_idx)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(input.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              ["original_image", "heat_map"],
                                              ["all", "absolute_value"],
                                              cmap=default_cmap,
                                              show_colorbar=True)

        return pred_label_idx


