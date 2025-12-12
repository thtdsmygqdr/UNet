
u_net_liver
===========
数据准备
-------
项目文件分布如下

  --project
  >main.py
  >>--data
  >>>--train
  >>>--val

模型训练
-------
完整UNet网络：
python main.py train --ablation=none
冻结深层：
python main.py train --ablation=freeze_deep
去除跳跃连接：
python main.py train --ablation=no_skip

测试模型训练
-----------
加载权重，默认保存最后一个权重

python main.py test --ablation=none --weight=weights_none_19.pth
python main.py test --ablation=freeze_deep --weight=weights_freeze_deep_19.pth
python main.py test --ablation=no_skip --weight=weights_no_skip_19.pth
每次测试会生成feature_visualization图像，对应测试集第一个样本在深层和浅层的表现