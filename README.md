# semantic-segmentation w/MRI
Final Project for Intro To Deep Learning ECGR 5106

Path for training a semantic segmentation model with encoder-decoder:
1. Pretrain CNN backbone (ConVNeXt) and Optimize
2. Construct the Encoder-Decoder Network.
3. Train the Encoder-Decoder Network
4. Evaluate
5. Inference

![image](https://github.com/Salitre97/semantic-segmentation-MRI/assets/126845001/cc443ac3-e937-4d66-97bc-b243564a4443)
Image Source: https://www.google.com/url?sa=i&url=https%3A%2F%2Fcnvrg.io%2Fsemantic-segmentation%2F&psig=AOvVaw3C6hrrIJRAKAsP_RVwhUMH&ust=1712790866826000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCLD4jq2htoUDFQAAAAAdAAAAABAE

***Classification vs Segmentation*** 
![image](https://github.com/Salitre97/semantic-segmentation-MRI/assets/126845001/03feb528-6a1e-4bc5-bded-6d91e4d2b064)

Image Source:
https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2021.638182/full


Encoder-Decoder Network Source:
https://towardsdatascience.com/understanding-u-net-61276b10f360

ConvNeXt Source:
https://github.com/yassouali/pytorch-segmentation/blob/8b8e3ee20a3aa733cb19fc158ad5d7773ed6da7f/models/segnet.py#L9

Pre-Processing Data for CNN Source:
https://towardsdatascience.com/how-to-apply-a-cnn-from-pytorch-to-your-images-18515416bba1

