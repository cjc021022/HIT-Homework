### 简介

这是智能无人机大作业的全部代码

详情可以参考这篇[blog](https://anlper.cn/2021/10/23/PX4%E6%97%A0%E4%BA%BA%E6%9C%BA-Gazebo%E4%BB%BF%E7%9C%9F%E5%AE%9E%E7%8E%B0%E7%A7%BB%E5%8A%A8%E7%89%A9%E4%BD%93%E7%9A%84%E8%B7%9F%E8%B8%AA/#%E7%A7%BB%E5%8A%A8%E5%B0%8F%E8%BD%A6%E7%9A%84%E5%AE%89%E8%A3%85)

### 文件功能

[cjc_test.cpp](./cjc_test.cpp) 是建立的ros节点，控制无人机飞行与跟踪

mavros_posix_sitl_cjc.launch 是gazebo仿真环境设置，主要涉及无人机和机器人turtlebot的融合等等

里面包含了深度摄像头的设置（depth_camera.sdf和posix_sitl_cp.launch)