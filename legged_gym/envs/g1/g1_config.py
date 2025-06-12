from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True     # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_dof_pos_limit = 0.9         # #DOF软限制：设置一个较小的活动范围，以防止 DOF 迅速达到极限，从而提高训练的稳定性
        base_height_target = 0.78        # 目标底座高度
        feet_height_target = 0.08        # 目标抬脚高度

        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            # tracking rewards
            tracking_goal_vel = 1.5
            tracking_yaw = 0.0
            tracking_ang_vel = 0.5   #奖励机器人在 z 轴上的角速度跟踪行为，从而减少不必要的旋转。
            lin_vel_z = -1.0         #机器人在Z轴方向上的线性速度进行惩罚（跳跃要改），uni是-2
            ang_vel_xy = -0.05       #该函数通过对角速度平方求和，惩罚机器人在 x 和 y 轴上的旋转运动(避免倾斜或侧翻），uni是-0.05
            orientation = -1.0       #惩罚机器人基础姿态（即机器人身体的朝向）偏离平坦的情况，uni是-1
            torques = -0.00001
            dof_vel = -1e-3          #惩罚关节速度，防止运动过快，uni是-1e-3
            dof_acc = -2.5e-7
            feet_air_time = 0.0      #对机器人脚部在空中的时间进行奖励,鼓励迈大步，uni是0，leg是5.0
            collision = -10          #惩罚机器人与环境中的物体发生碰撞
            feet_stumble = -1.0      #惩罚机器人脚部绊倒
            action_rate = -0.01
            stand_still = -0.
            dof_pos_limits = -5.0    #对接近关节位置极限的情况进行惩罚，uni是-5
            # 下面的奖励项是g1中的
            alive = 0.15             #奖励机器人处于活跃状态
            hip_pos = -1.0           #惩罚机器人的髋关节当前位置与默认位置的偏差，uni是-1（跟踪了两个髋关节，防止腿旋转）
            contact_no_vel = -0.2    #惩罚脚无速度接触地面，使走路更平滑
            contact = 0.18           #奖励脚部的接触状态与支撑态状态一致情况
            no_fly = 0.0             #奖励机器人正好有一个脚在地面上，防止下楼飞行

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        num_steps_per_env = 24 # per iteration
        max_iterations = 30000
        run_name = ''
        experiment_name = 'g1'
        save_interval = 1000 # check for potential saves every this many iterations