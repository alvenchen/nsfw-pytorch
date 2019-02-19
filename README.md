NSFW pytorch implementation
=========
This repo contains code for running Not Suitable for Work (NSFW) classificationdeep neural network pytorch models.

According to [yahoo-open-nsfw](https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for)
fine-tune with resnet

Data
---------
You can get some data from [nsfw_data_scrapper](https://github.com/alexkimxyz/nsfw_data_scrapper).
Then put your datas at data/train and data/test.

Train
---------
Get [pretrained resnet model](https://download.pytorch.org/models/resnet50-19c8e357.pth), and put it in models/.

start trainning:
```bash
python src/train.py
```
