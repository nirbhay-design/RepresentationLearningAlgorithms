## **AIM of the Project**

- To implement Representation Learning algorithms (Contrastive and self-supervised learning algorithms)

## **Algos Implemented**

- [x] [SimCLR (ICML 2020)](https://arxiv.org/pdf/2002.05709)
- [x] [SupCon (NIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf)
- [x] [SimSiam (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf)
- [x] [Bootstrap your own Latent (BYOL) (NIPS 2020)](https://arxiv.org/pdf/2006.07733)
- [x] [Barlow Twins (ICML 2021)](https://arxiv.org/pdf/2103.03230)
- [x] [TripletMarginLoss (CVPR 2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)

<!-- - [ ] [Momentum Contrast (MoCo) (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf) -->


## **Results**

|Algorithm|CIFAR10|CIFAR100|
|---|---|---|
|SimCLR|87.5|57.7|
|SupCon|94.0|74.7|
|Triplet|83.4|76.3|
|Barlow Twins|81.2|47.7|
|BYOL|83.0|47.0|
|SimSiam|-|-|
|Bhattacharya(Ours)|63.9|32.2|

## **Extension**

- Knowledge Distillation Regularizer

## **Team Members**

- Mansi Tomer (SR: 24014)
- Nirbhay Sharma (SR: 24806)