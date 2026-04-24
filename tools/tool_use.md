不需要 GPU。这个脚本全是 CPU 工作：

  - 读 laz/json.gz
  - 做坐标变换
  - numpy.histogramdd 栅格化
  - 写图片和 JSON

  所以你不用专门申请 GPU，普通 CPU 节点就能跑。真正要注意的是内存和 IO，但这个脚本默认一次只处理单条 route 的少量帧，压力不大。

  我已经把批量模式补进去了，脚本还是同一个：carla_garage/tools/render_lidar_bev_alignment.py:1。

  单 route 跑法

  cd /home/heavenlysu/sitp_workspace/carla_garage

  python tools/render_lidar_bev_alignment.py \
    --route-dir /path/to/one/route_dir \
    --history 1 2 3 \
    --output-dir /tmp/lidar_bev_debug

  批量跑法
  给一个上层目录，它会递归找所有同时含有 lidar/ 和 measurements/ 的 route 目录：

  python tools/render_lidar_bev_alignment.py \
    --root-dir /home/heavenlysu/sitp_workspace/carla_garage/data/50x36_Town13 \
    --history 1 2 3 \
    --frames-per-route 2 \
    --max-routes 10 \
    --output-dir /tmp/lidar_bev_batch

  常用参数

  - --frames-per-route 2
    每条 route 选 2 个有足够历史帧的时刻。
  - --max-routes 10
    先只抽 10 条 route，避免一次跑太大。
  - --frame 120
    指定固定 frame；一旦指定，就每条 route 都只尝试这个 frame。
  - --include-ground-plane
    如果你想看双通道 BEV 的第二层逻辑，可以开；默认只看上层通道。

  输出结构

  /tmp/lidar_bev_batch/
    summary.json
    Town13__Scenario__Route.../
      frame_0120/
        history_01.png
        history_02.png
        history_03.png
      frame_0240/
        ...

  怎么看结果
  终端会打印每组：

  ... before_iou=0.12 after_iou=0.37 gain=0.25

  正常应该是 after_iou > before_iou。图里：

  - 左：当前帧
  - 中：未对齐历史帧叠图
  - 右：对齐后历史帧叠图

  如果你愿意，我下一步可以继续给你补一个“自动筛选最值得看的失败样本”的汇总脚本，比如按 iou_gain 最低的 route/frame 排序。