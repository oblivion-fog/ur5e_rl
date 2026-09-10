# NOTICE

本工程为 UR5e 末端视觉追踪 + 真空吸附抓取强化学习实验工程 v5 Flow-first。

v5 针对 v4 的 `pregrasp=1 / ready=0 / contact=0 / success=0` 局部最优进行了重构：
取消目标相关强 Cartesian shield，允许真空提前开启，reward 改为 progress + 一次性技能事件，Stage1 目标放到吸盘正下方附近，并固定适度 SAC entropy 以保持探索。

仍保留机器人本体撞桌/撞地、非吸盘部位推 cube 的安全终止，以避免重新出现旧版本的碰撞作弊行为。

本包不包含 MuJoCo Menagerie UR5e 的 OBJ mesh 资产；请使用你已有的 UR5e `assets/` 文件夹或按 `scripts/prepare_assets.py` 说明准备。
