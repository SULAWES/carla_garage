我的笔记喵

made by claude & reviewed by codex

LiDAR逻辑已修改为sensor_agent同款，codex提醒对当前 CARLA 半帧 LiDAR 设定这是合理的，但会让前几步一直刹车，不是 bug，只是要确认这和你训练/评测预期一致。

这里你用的是 self.state_log[-(i + 1)] 去取历史姿态，而原版 carla_garage/team_code/sensor_agent.py:476 用的是 self.state_log[i]。从时序对应关系上看，你现在这版反而更像是“按 buffer 的相对时间正确对齐”，我不认为这是问题，但它和原版不完全一致，最好在可视化里确认一下历史 LiDAR 重投影后是否真的更稳定。

ability_data.py 主要给我们的启发

ability_data.py 的重点不是 extra_sensors 本身，而是按能力/场景筛数据：

- Overtaking
- Merging
- Emergency_Brake
- Give_Way
- Traffic_Sign
- No_Scenario

见 syb_carla_garage/team_code/ability_data.py:75。

这对后续 DiffusionDrive 很有价值，但它属于 数据采样/评估分桶，不是 status_feature 的输入维度问题。

所以最终可以这样记：

extra_sensors 旧名：garage/syb 旧模型里的 7 维 speed+command 条件 token
status_feature 当前名：DiffusionDrive 的低维条件 token

推荐做法：DD 保留 status_feature 这个入口，但把内容改成 7 维 speed+command。

这也解释了你记得的“10 维迁移到 6(+1) 维”：这个事项确实应该做，而且来源就是 garage/syb 这条线的 extra_sensors 设计。


command时序差异出自对garage sensor_agent的继承


先用CPU缓存数据，然后再申请GPU训练。


开环效果好不代表闭环好

num_workers大一点。

实验发现，7 CPU物理核心的时候num-workers设置为16有效（

haonimaduobishi
caonima

传感器
1024*512 1024*384 1024*256
LiDAR
SpeedHead
额外的两个condition token
ResNet34 vs ResNet50

图像输入改成1024*384

ImageNet normalization开关

修正 B2D source image size 记录

manifest / dataset 加 skip-first参数
    --skip-first-frames 25

SpeedHead-v1
- 数据层已经有：

    target_speed
    brake
    target_speed_twohot
    target_speed_class
    target_speed_label_valid

- 还要补：

    model speed head
    speed loss
    speed metrics
    training_config 记录

推理侧 predicted-speed controller 开关

Route condition token
