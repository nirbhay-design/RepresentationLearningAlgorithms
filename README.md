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

|Algorithm|CIFAR10 (R50)|CIFAR100 (R50)|CIFAR10 (R18)|CIFAR100 (R18)|
|---|---|---|---|---|
|SimCLR|87.5|57.7|85.9|55.0|
|SupCon|**94.0**|74.7|**93.5**|**70.4**|
|Triplet|83.4|**76.3**|86.0|64.5|
|Barlow Twins|81.2|47.7|80.3|45.8|
|BYOL|83.0|47.0|84.8|54.8|
|SimSiam|76.5|34.5|88.6|62.3|
|DARe (Ours)|-|-|87.3|-|

## **Extension**

- Distribution Alignment Regularized (DARe) contrastive learning 

$$
\begin{align}
\mu_1, \log(\sigma_1^2) = f(x_1) \\
\mu_2, \log(\sigma_2^2) = f(x_2) \\
z_1 = \mu_1 + \epsilon_1 \odot \sigma_1 \\
z_2 = \mu_2 + \epsilon_2 \odot \sigma_2 \\
\mathcal{L} = \mathcal{L}_{\mathcal{CON}} + \lambda \mathcal{L}_{\mathcal{DAL}} \\
\mathcal{L}_{\mathcal{CON}} = \sum_{} -log(\frac{e^{sim(z_{1i}, z_{2i})}}{\sum_{j \ne i} e^{sim(z_{1i}, z_{2j})}}) \\
\mathcal{L}_{\mathcal{DAL}} = \mathcal{JSD}(\mathcal{N}(\mu_1, \sigma_1^2) || \mathcal{N}(\mu_2, \sigma_2^2))
\end{align}
$$


## **Team Members**

- Mansi Tomer (SR: 24014)
- Nirbhay Sharma (SR: 24806)