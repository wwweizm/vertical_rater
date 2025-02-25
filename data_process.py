import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "trajectory.csv"

def calculate_velocity(data, x_column, z_column):

    velocity_x = data[x_column].diff() 
    velocity_z = data[z_column].diff() 

    data['velocity_x'] = velocity_x
    data['velocity_z'] = velocity_z
    # 返回速度数组
    return data[['velocity_x', 'velocity_z']].values

try:
    data = pd.read_csv(file_path)
    print("successfully loaded file")
except Exception as e:
    print(f"failed to load file: {e}")
    exit()

# 添加时间列
data['time'] = np.arange(len(data))

# 检查所需的列是否存在
required_columns = ['body_left_hip_x', 'body_right_hip_x', 'body_left_hip_z', 'body_right_hip_z']
for col in required_columns:
    if col not in data.columns:
        print(f"缺少列：{col}")
        exit()

# 计算中点轨迹
data['midpoint_x'] = (data['body_left_hip_x'] + data['body_right_hip_x']) / 2
data['midpoint_z'] = (data['body_left_hip_z'] + data['body_right_hip_z']) / 2

data['upper_x'] = (data['body_left_shoulder_x'] + data['body_right_shoulder_x']) / 2
data['upper_z'] = (data['body_left_shoulder_z'] + data['body_right_shoulder_z']) / 2

midpoint_trajectory = data[['midpoint_x', 'midpoint_z']].values

#print(midpoint_trajectory)

midpoint_velocity = calculate_velocity(data, 'midpoint_x', 'midpoint_z')

#print(midpoint_velocity)
def plot_mid_point(data):
    plt.figure(figsize=(10, 6))

    # 绘制轨迹
    plt.plot(data[:, 0], data[:, 1], label='Trajectory', marker='o', alpha=0.8)

    # 图表细节
    plt.title("Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Z Position")
    plt.legend()
    plt.grid()
    plt.show()

plot_mid_point(midpoint_trajectory)

def plot_velocity(data, velocity):
    plt.figure(figsize=(10, 6))

    # 绘制轨迹
    plt.plot(data['time'] ,velocity [:, 0], label='velocity_x', alpha=0.8)
    plt.plot(data['time'] ,velocity [:, 1], label='velocity_z', alpha=0.8)
    # 图表细节
    plt.title("Trajectory")
    plt.xlabel("time")
    plt.ylabel("velocity")
    plt.legend()
    plt.grid()
    plt.show()

plot_velocity(data, midpoint_velocity)

def calculate_unit_kinetic_energy(velocity_x,velocity_z):
    max_v_x_before_index=0
    max_v_x_after_index=0
    max_v_z_index,min_v_z_index = 0,0
    max_v_z = 0
    min_v_z = 999999999
    max_v_x_before = 0    
    max_v_x_after = 0
    if velocity_x[1] < 0 :
        velocity_x = -velocity_x
    for i in range (len(velocity_z)):
        if velocity_z[i] > max_v_z:
            max_v_z = velocity_z[i]
            max_v_z_index = i

    for i in range (len(velocity_x)):

        if velocity_x[i] > max_v_x_before and i < max_v_z_index:
            max_v_x_before = velocity_x[i]
            max_v_x_before_index = i
        if velocity_x[i] > max_v_x_after and i > max_v_z_index:
            max_v_x_after = velocity_x[i]
            max_v_x_after_index = i
        if velocity_z[i] < min_v_z and i < max_v_z_index:
            min_v_z = velocity_z[i]
            min_v_z_index = i

    
    # 计算最大正向速度并求平方
    max_positive_velocity = velocity_x[max_v_z_index]
    k_x_before = max_v_x_before ** 2
    k_x_after = max_v_x_after ** 2
    k_z = max_positive_velocity ** 2
    
    return k_x_before, k_x_after, k_z, min_v_z, max_v_z_index, min_v_z_index, max_v_x_before_index, max_v_x_after_index

k_x_before,k_x_after,k_z,min_v_z,max_v_z_index, min_v_z_index, max_v_x_before_index, max_v_x_after_index= calculate_unit_kinetic_energy(midpoint_velocity[:, 0],midpoint_velocity[:, 1])

run_up_quality = k_x_before / k_z
brake_quality = k_x_before / k_x_after
high_jump_quality = k_x_after / k_z

print(  f"run_up_quality_ori: {run_up_quality}\n"  )
print(  f"brake_quality_ori: {brake_quality}\n"  )
print(  f"high_jump_quality_ori: {high_jump_quality}\n"  )

def score_run_up_quality(value):
    score = 100 * (1-(0.6-value)**2/25)
    if score < 50:
        return 50
    else:
        return score


def score_brake_quality(value):
    score = 100 * (1-(0.6-value)**2/25)
    if score < 50:
        return 50
    else:
        return score


def score_high_jump_quality(value):
    score = 100 * (1-(0.6-value)**2/25) 
    if score < 50:
        return 50
    else:
        return score 

run_up_quality = score_run_up_quality(run_up_quality)
brake_quality = score_brake_quality(brake_quality)
high_jump_quality = score_high_jump_quality(high_jump_quality)

print(  f"run_up_quality: {run_up_quality}\n"  )
print(  f"brake_quality: {brake_quality}\n"  )
print(  f"high_jump_quality: {high_jump_quality}\n"  )

def calculate_upper_body_angle(data, lower_x_col, lower_z_col, upper_x_col, upper_z_col):
    """
    计算上半身的角度
    参数:
        data (pd.DataFrame): 包含髋部和肩部坐标数据的DataFrame
        lower_x_col (str): 髋部中点X坐标列名
        lower_z_col (str): 髋部中点Z坐标列名
        upper_x_col (str): 肩部中点X坐标列名
        upper_z_col (str): 肩部中点Z坐标列名
    返回:
        pd.Series: 上半身角度，单位为度
    """
    # 计算肩部和髋部的坐标差
    delta_z = data[upper_z_col] - data[lower_z_col]
    delta_x = data[upper_x_col] - data[lower_x_col]
    
    # 计算角度（弧度）
    angles_rad = np.arctan2(delta_z, delta_x)  
    angles_deg = np.degrees(angles_rad)
    
    return angles_deg

data['upper_body_angle'] = calculate_upper_body_angle(
    data, 
    lower_x_col='midpoint_x', 
    lower_z_col='midpoint_z', 
    upper_x_col='upper_x', 
    upper_z_col='upper_z'
)

def plot_upper_body_angle(angle):
    plt.figure(figsize=(10, 6))

    # 绘制轨迹
    plt.plot(data['time'], abs(angle), label='Trajectory', marker='o', alpha=0.8)

    # 图表细节
    plt.title("Upper Body Angle")
    plt.xlabel("Time")
    plt.ylabel("Upper body angle")
    plt.legend()
    plt.grid()
    plt.show()

plot_upper_body_angle(data['upper_body_angle'].values)

def score_upper_body_angle(angle):
    angle = abs(angle)
    max_upper_body_angle = 0
    max_upper_body_angle_index = 0
    for i in range (max_v_z_index):
        if (abs(data['upper_body_angle'][i]-90) > max_upper_body_angle) and i>=max_v_x_before_index-20: 
            max_upper_body_angle = abs(angle[i]-90)
            max_upper_body_angle_index = i
    score = 100 * (1 - (max_upper_body_angle-45)**2 / 45**2)
    return max_upper_body_angle,max_upper_body_angle_index,score

max_upper_body_angle,max_upper_body_angle_index,upper_body_angle_score = score_upper_body_angle(data['upper_body_angle'].values)
print('max_upper_body_angle_score:',upper_body_angle_score)
print('max_upper_body_angle:',max_upper_body_angle)

def calculate_knee_angles(data):
    """
    计算左右膝盖的角度，并返回左右膝盖的最大角度及对应的索引。
    
    参数:
        data (pd.DataFrame): 包含髋、膝和踝关节坐标的DataFrame，列名为
                             'body_left_hip_x', 'body_left_hip_z',
                             'body_left_knee_x', 'body_left_knee_z',
                             'body_left_ankle_x', 'body_left_ankle_z',
                             'body_right_hip_x', 'body_right_hip_z',
                             'body_right_knee_x', 'body_right_knee_z',
                             'body_right_ankle_x', 'body_right_ankle_z'.
    """
    def calculate_angle(hip_x, hip_z, knee_x, knee_z, ankle_x, ankle_z):
        """
        计算膝关节的角度。
        """
        # 髋到膝
        vector1 = np.array([hip_x - knee_x, hip_z - knee_z])
        # 踝到膝
        vector2 = np.array([ankle_x - knee_x, ankle_z - knee_z])
        
        # 计算两个向量的夹角
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        # 角度公式：arccos(dot_product / (norm1 * norm2))
        cos_theta = dot_product / (norm1 * norm2)
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 防止数值误差
        
        # 返回角度（以度为单位）
        return np.degrees(angle_rad)

    # 计算左膝角度
    data['left_knee_angle'] = data.apply(
        lambda row: calculate_angle(
            row['body_left_hip_x'], row['body_left_hip_z'],
            row['body_left_knee_x'], row['body_left_knee_z'],
            row['body_left_ankle_x'], row['body_left_ankle_z']
        ), axis=1
    )
    
    # 计算右膝角度
    data['right_knee_angle'] = data.apply(
        lambda row: calculate_angle(
            row['body_right_hip_x'], row['body_right_hip_z'],
            row['body_right_knee_x'], row['body_right_knee_z'],
            row['body_right_ankle_x'], row['body_right_ankle_z']
        ), axis=1
    )

    # 找到满足条件的最小左膝角度及其索引
    left_knee_subset = data.loc[max_v_x_before_index-5:max_v_z_index+5, 'left_knee_angle']
    min_left_knee_angle = left_knee_subset.min()
    min_left_knee_index = left_knee_subset.idxmin()

    # 找到满足条件的最小右膝角度及其索引
    right_knee_subset = data.loc[max_v_x_before_index-5:max_v_z_index+5, 'right_knee_angle']
    min_right_knee_angle = right_knee_subset.min()
    min_right_knee_index = right_knee_subset.idxmin()

    now_index = 0
    #刹车角度计算
    if min_left_knee_angle < min_right_knee_angle:
        now_index = min_left_knee_index-1
        knee_angle_cat = data['left_knee_angle'].values
    else:   
        now_index = min_right_knee_index-1
        knee_angle_cat = data['right_knee_angle'].values

    brake_knee_angle = 0
    for i in range (len(knee_angle_cat)):
        if knee_angle_cat[now_index-i-1] >= knee_angle_cat[now_index-i]:
            brake_knee_angle = knee_angle_cat[now_index-i-1]
            i = i-1
        else:
            break

    # 返回结果
    return {
        'min_left_knee_angle': min_left_knee_angle,
        'min_left_knee_index': min_left_knee_index,
        'min_right_knee_angle': min_right_knee_angle,
        'min_right_knee_index': min_right_knee_index,
        'brake_knee_angle': brake_knee_angle,
    }

result_knee = calculate_knee_angles(data)

def score_knee_angle():
    if result_knee['min_left_knee_angle'] < result_knee['min_right_knee_angle']:
        knee_angle_large = result_knee['min_left_knee_angle']
        knee_angle_small = result_knee['min_right_knee_angle']
    else:
        knee_angle_large = result_knee['min_right_knee_angle']
        knee_angle_small = result_knee['min_left_knee_angle']
    score_l= 100 * (1 - 0.25*(knee_angle_large-90)**2 / 30**2)
    score_s= 100 * (1 - 0.25*(knee_angle_small-120)**2 / 40**2)
    score_brake_knee_angle = 100 * (1 - 0.25*(result_knee['brake_knee_angle']-170)**2 / 20**2)
    if score_brake_knee_angle < 30:
        score_brake_knee_angle = 30
    elif result_knee['brake_knee_angle']>160:
        score_brake_knee_angle = 100
    if knee_angle_large <=45 or knee_angle_large >= 150:
        score_l = 30
    if knee_angle_small <=60 or knee_angle_small == 180:
        score_s = 30
    return score_l,score_s,score_brake_knee_angle,score_l/2+score_s/2    
score_knee_angle_l,score_knee_angle_s,score_brake_knee_angle,knee_angle_score = score_knee_angle()
def plot_knee_angle():
    plt.figure(figsize=(10, 6))

    # 绘制轨迹
    plt.plot(data['time'], abs(data['right_knee_angle'].values), label='right_knee_angle', marker='o', alpha=0.8)
    plt.plot(data['time'], abs(data['left_knee_angle'].values), label='left_knee_angle', marker='s', alpha=0.8)

    # 图表细节
    plt.title("Knee Angle")
    plt.xlabel("Time")
    plt.ylabel("Knee angle")
    plt.legend()
    plt.grid()
    plt.show()

plot_knee_angle()    
# 输出最大角度和对应索引
print(f"max_left_knee_angle: {result_knee['min_left_knee_angle']}")
print(f"max_right_knee_angle: {result_knee['min_right_knee_angle']}")
print(f"brake_knee_angle: {result_knee['brake_knee_angle']}")
print(f"score_large_knee_angle: {score_knee_angle_l}")
print(f"score_small_knee_angle: {score_knee_angle_s}")
print(f"score_brake_knee_angle: {score_brake_knee_angle}")


def calculate_arm_angle(data):
    
    # 计算肩部和髋部的坐标差
    delta_z_left = data['left_hand_z'] - data['body_left_shoulder_z']
    delta_x_left = data['left_hand_x'] - data['body_left_shoulder_x']
    delta_z_right = data['right_hand_z'] - data['body_right_shoulder_z']
    delta_x_right = data['right_hand_x'] - data['body_right_shoulder_x']

    # 计算角度（弧度）
    angles_rad = np.arctan2(delta_z_left, delta_x_left)  
    angles_deg_left = np.degrees(angles_rad)
    
    angles_rad = np.arctan2(delta_z_right, delta_x_right)
    angles_deg_right = np.degrees(angles_rad)

    data['left_arm_angle'] = angles_deg_left
    data['right_arm_angle'] = angles_deg_right

    return    


calculate_arm_angle(data)

def plot_arm_angle():
    plt.figure(figsize=(10, 6))

    # 绘制轨迹
    plt.plot(data['time'], abs(data['right_arm_angle'].values), label='right_arm_angle', marker='o', alpha=0.8)
    plt.plot(data['time'], abs(data['left_arm_angle'].values), label='left_arm_angle', marker='s', alpha=0.8)

    # 图表细节
    plt.title("Arm Angle")
    plt.xlabel("Time")
    plt.ylabel("Arm angle")
    plt.legend()
    plt.grid()
    plt.show()

plot_arm_angle()

import numpy as np

def calculate_arm_swing_velocity(data):
    """
    计算左手臂的摆臂角速度。

    参数:
        data (pd.DataFrame): 包含左手臂角度的DataFrame，列名包含：
                             'left_arm_angle'（和右手角度可以类比）.
    
    """

    # 找到角度最大值的索引
    max_left_arm_angle_index = data['left_arm_angle'].idxmax()
    
    # 从最大角度处开始向前搜索直到手臂角度增大
    left_arm_start_index = 0
    for i in range(max_left_arm_angle_index):
        if data['left_arm_angle'][max_left_arm_angle_index-i-1] < data['left_arm_angle'][max_left_arm_angle_index-i]:
            left_arm_start_index = i
        else:
            break
        
    # 计算角度差和索引差并计算摆臂的角速度
    angle_diff = data['left_arm_angle'][max_left_arm_angle_index] - data['left_arm_angle'][left_arm_start_index]
    index_diff = max_left_arm_angle_index - left_arm_start_index
    
    # 避免除以0错误
    if index_diff != 0:
        left_arm_swing_velocity = angle_diff / index_diff
    else:
        left_arm_swing_velocity = 0
    
    max_right_arm_angle_index = data['right_arm_angle'].idxmax()
    #print(f"最大右手臂角度的索引: {max_right_arm_angle_index}")
    right_arm_start_index = 0
    for i in range(max_right_arm_angle_index):
        if data['right_arm_angle'][max_right_arm_angle_index-i-1] < data['right_arm_angle'][max_right_arm_angle_index-i]:
            right_arm_start_index = i
        else:
            break
   
    # 计算角度差和索引差并计算摆臂的角速度
    angle_diff = data['right_arm_angle'][max_right_arm_angle_index] - data['right_arm_angle'][right_arm_start_index]
    index_diff = max_right_arm_angle_index - right_arm_start_index
    
    # 避免除以0错误
    if index_diff != 0:
        right_arm_swing_velocity = angle_diff / index_diff
    else:
        right_arm_swing_velocity = 0
    

    # 存储角速度
    return left_arm_swing_velocity, right_arm_swing_velocity, left_arm_start_index, max_left_arm_angle_index, right_arm_start_index, max_right_arm_angle_index

left_arm_swing_velocity, right_arm_swing_velocity, left_arm_start_index, max_left_arm_angle_index, right_arm_start_index, max_right_arm_angle_index = calculate_arm_swing_velocity(data)

def score_arm_swing_velocity(left_arm_swing_velocity, right_arm_swing_velocity):
    score_left = 100 * (1 - (abs(left_arm_swing_velocity)-15)**2 / 10**2)
    score_right = 100 * (1 - (abs(right_arm_swing_velocity)-15
                              )**2 / 10**2)
    final_score = score_left/2+score_right/2
    if final_score < 50:
        return 50
    else:
        return final_score

arm_swing_score = score_arm_swing_velocity(left_arm_swing_velocity, right_arm_swing_velocity)

print(f"left_arm_omega: {left_arm_swing_velocity}")
print(f"right_arm_omega: {right_arm_swing_velocity}")

def grade_score(score):
    """根据评分返回评级"""
    if score >= 80:
        return 'A'
    elif score >= 60:
        return 'B'
    elif score >= 40:
        return 'C'
    else:
        return 'D'

def improvement_suggestion(grade, label):
    """根据评级给出改进建议，针对不同的指标"""
    suggestions = {
        'Run-up Quality': {
            'A': "Run-up quality is excellent! Continue maintaining your speed and form.",
            'B': "Work on your acceleration and stride efficiency to improve your run-up quality.",
            'C': "Consider optimizing your approach and focus on smooth acceleration during the run-up.",
            'D': "Revise your approach technique and practice a consistent stride pattern."
        },
        'Brake Quality': {
            'A': "Brake quality is solid! Maintain control and ensure smooth deceleration.",
            'B': "Improve your deceleration technique by focusing on gradual and controlled braking.",
            'C': "Work on your stopping technique. Consider practicing deceleration drills to improve your control.",
            'D': "Significant improvement needed in your brake quality. Focus on mastering the stop sequence."
        },
        'High Jump Quality': {
            'A': "Excellent high jump! Keep your posture and angle in check during the ascent.",
            'B': "Increase your jump height by working on your take-off angle and technique.",
            'C': "Review your take-off technique and focus on generating more upward force.",
            'D': "Your jump needs improvement. Practice explosive jumps and focus on your posture."
        },
        'Upper Body Angle': {
            'A': "Upper body angle is ideal! Keep up the good form.",
            'B': "Work on improving your upper body posture to enhance your jump efficiency.",
            'C': "Focus on refining your posture and timing during the take-off phase.",
            'D': "Upper body posture needs significant attention. Work on aligning your body during the jump."
        },
        'Knee Angle': {
            'A': "Knee angle is optimal! Maintain your joint flexibility and technique.",
            'B': "Improve knee angle during the jump to ensure proper form and force transfer.",
            'C': "Work on your knee angle consistency and focus on controlled movement.",
            'D': "Knee angle is a critical area. Focus on proper knee joint positioning during take-off and jump."
        },
        'Brake Knee Angle': {
            'A': "Brake knee angle is great! Continue refining your knee control during braking.",
            'B': "Ensure smoother control of your knee during deceleration to reduce braking time.",
            'C': "Practice more controlled deceleration with an emphasis on proper knee joint technique.",
            'D': "Brake knee angle requires significant improvement. Focus on joint mobility and braking techniques."
        },
        'Arm Swing Velocity': {
            'A': "Arm swing velocity is excellent! Your arm movement is helping you jump higher.",
            'B': "Improve your arm swing technique to generate more upward force.",
            'C': "Focus on optimizing your arm movement for a more efficient jump.",
            'D': "Your arm swing is lacking. Work on timing and coordination to improve your arm velocity."
        }
    }

    # 获取具体的指标的建议
    if label in suggestions:
        suggestion_dict = suggestions[label]
        return suggestion_dict[grade]
    else:
        return "No suggestion available for this metric."


def plot_radar_chart(scores, labels, title="Radar Chart",report_text=None):
    """
    参数:
        scores (list or np.ndarray): 需要绘制的分数列表
        labels (list): 每个分数对应的标签
        title (str): 图表标题
    """
    # 数据准备
    num_vars = len(scores)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    scores += scores[:1]  # 闭合数据
    angles += angles[:1]  # 闭合角度

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='blue', alpha=0.25)
    ax.plot(angles, scores, color='blue', linewidth=2)
    
    # 添加标签和刻度
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color="gray", fontsize=10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    
    # 图表标题
    ax.set_title(title, size=15, color='blue', y=1.1)

    if report_text:
        plt.figtext(0.1, -0.2, report_text, wrap=True, horizontalalignment='left', fontsize=10, color='black')

    # 显示雷达图
    plt.show()

scores = [ run_up_quality, brake_quality, high_jump_quality, upper_body_angle_score, knee_angle_score, score_brake_knee_angle, arm_swing_score] 
labels = ["Run-up Quality", "Brake Quality", "High Jump Quality", "Upper Body Angle", "Knee Angle","Brake Knee Angle", "Arm Swing Velocity"]

def print_scores_and_ratings(scores, labels):
    """打印每个项目的评分和评级"""
    print("Scores and Ratings:")
    for score, label in zip(scores, labels):
        grade = grade_score(score)
        print(f"{label}: {score} -> {grade}")

def print_improvement_suggestions(scores, labels):
    """根据评分打印改进建议"""
    print("\nSuggestion:")
    for score, label in zip(scores, labels):
        grade = grade_score(score)
        suggestion = improvement_suggestion(grade, label)
        print(f"- {label}: {suggestion}")

report_text = """
Performance Evaluation Report:

1. Run-up Quality: The run-up quality is assessed based on speed and form during the approach.
2. Brake Quality: This metric reflects the ability to decelerate and stop smoothly.
3. High Jump Quality: Focuses on the efficiency of the jump phase, including peak height and angle.
4. Upper Body Angle: Evaluates the angle of the upper body at key moments of the jump.
5. Knee Angle: The knee angle at different phases of the jump provides insight into joint efficiency.
6. Brake Knee Angle: This score evaluates knee joint motion during deceleration.
7. Arm Swing Velocity: Measures the velocity of the arm swing to assist in jump height.

The scores are a summary of these metrics, with higher scores reflecting better performance.
"""

# 绘制雷达图并添加报告文本
plot_radar_chart(scores, labels, title="Performance Evaluation", report_text=report_text)

# 输出各个项目的评分和对应的评级
print_scores_and_ratings(scores, labels)

# 输出改进建议
print_improvement_suggestions(scores, labels)

# 绘制雷达图
#plot_radar_chart(scores, labels, title="Performance Evaluation")


