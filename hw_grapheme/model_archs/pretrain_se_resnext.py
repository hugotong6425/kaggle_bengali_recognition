from torch import nn
import pretrainedmodels


class se_resnext50_pretrained(nn.Module):
    def __init__(self, head):
        super(se_resnext50_pretrained, self).__init__()
        model_name = 'se_resnext50_32x4d'
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

        self.layer0 = model.layer0
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
        last_layer_in_feature = model.last_linear.in_features
        self.head_root = head(last_layer_in_feature, 168)
        self.head_vowel = head(last_layer_in_feature, 11)
        self.head_consonant = head(last_layer_in_feature, 7)

    def forward(self, x):
        # print(x.shape)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # print(x.shape)
        x_root = self.head_root(x)
        x_vowel = self.head_vowel(x)
        x_consonant = self.head_consonant(x)

        return x_root, x_vowel, x_consonant