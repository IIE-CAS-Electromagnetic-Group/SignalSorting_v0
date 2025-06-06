'''
相比较上一个版本引入了基于 softmax 权重机制的奖励函数自学习框架。
与原始版本中手动设定 reward 函数各项权重不同，当前实现将 reward 函数拆解为多个可解释分量（如得分提升项、边缘增益项等），
并通过引入可学习的权重参数，在训练过程中自动调整各项权重组合。
该机制无需手动调参，通过联合优化策略网络和 reward 权重，使奖励函数能自适应地引导策略朝向更优的锚框调整方向。
此外，权重通过 softmax 归一化以防止 reward hacking，并支持权重保存与加载，便于模型迁移与复现。
'''
import datetime

import pandas as pd

import time
from joblib import Parallel, delayed


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np




from utils.dataset_preprocessing import df_normalization
from utils.plot_greyscale import plot_greyscale_for_singledf, plot_greyscale_for_singledf_with_anthor, save_as_gif, \
    plot_greyscale_for_singledf_with_anthor_and_score, plot_greyscale_for_singledf_with_anthor_and_mark
from feature_extraction import get_feature, train_model
from adjustment_action import adjust_anchor_box
from calculate_reward import calculate_reward_new, calculate_score
from init_anchor import generate_initial_anchors, merge_anchor


class SharedFeatureExtractor(nn.Module):
    def __init__(self, region_dim=64, anchor_dim=4, embed_dim=512):
        super().__init__()
        self.region_fc = nn.Linear(region_dim, embed_dim // 2)
        self.anchor_fc = nn.Linear(anchor_dim, embed_dim // 2)
        self.fc_combined = nn.Linear(embed_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=2048)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.output_norm = nn.LayerNorm(embed_dim)

        # 初始化权重（Xavier）
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        r = self.region_fc(x[:, :64])
        a = self.anchor_fc(x[:, 64:])

        x_region = x[:, :64]
        if torch.isnan(x_region).any() or torch.isinf(x_region).any():
            print("region_fc 输入含 NaN/Inf！")
            print(x_region)

        if torch.isnan(r).any() or torch.isinf(r).any():
            print("region_fc 输出异常！", r)
        if torch.isnan(a).any() or torch.isinf(a).any():
            print("anchor_fc 输出异常！", a)

        combined = torch.cat([r, a], dim=1)
        combined = self.fc_combined(combined)

        if torch.isnan(combined).any() or torch.isinf(combined).any():
            print("fc_combined 输出异常！", combined)


        seq = combined.unsqueeze(0)  # [1, B, D]
        out = self.transformer(seq).squeeze(0)  # [B, D]
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("transformer 输出异常！")

        out = self.output_norm(out)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("output_norm 输出异常！")


        return out


class PPOActor(nn.Module):
    def __init__(self, feature_extractor, action_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.policy_head = nn.Linear(512, action_dim)

    def forward(self, x):
        features = self.feature_extractor(x)

        if torch.isnan(features).any() or torch.isinf(features).any():
            print("Feature extractor输出非法值（NaN/Inf）")
            #features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        logits = self.policy_head(features)
        logits = torch.clamp(logits, -10, 10)  # 防止softmax不稳定

        if torch.isnan(logits).any():
            print("logits 出现 NaN，")
            #logits = torch.zeros_like(logits)

        probs = F.softmax(logits, dim=-1)

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("softmax 后出现 NaN/Inf，")
            #probs = torch.full_like(probs, 1.0 / probs.shape[-1])

        return probs, logits

class PPOCritic(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        value = self.value_head(features).squeeze(-1)
        value = torch.clamp(value, -1000, 1000)  # 限制 critic 输出范围
        return value



class PPOAgent:
    def __init__(self, input_dim, action_space, device='cpu', clip_eps=0.2, gamma=0.99, lam=0.95):
        self.device = device
        self.action_space = action_space
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam

        self.feature_extractor = SharedFeatureExtractor().to(device)
        self.actor = PPOActor(self.feature_extractor, len(action_space)).to(device)
        self.critic = PPOCritic(self.feature_extractor).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=5e-5
        )

        # Learnable reward weights (6维), softmax 使用
        init_weights = torch.log(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float))
        self.reward_weights = nn.Parameter(init_weights, requires_grad=True)
        self.reward_optimizer = torch.optim.Adam([self.reward_weights], lr=5e-5)

        self.buffer = []
        self.buffer_reward_components = []  # 存储 (r1, r2)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        torch.save(self.reward_weights, os.path.join(path, 'reward_weights.pt'))
        print(f"Model and reward weights saved to {path}")

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth'), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth'), map_location=self.device))
        self.reward_weights = torch.load(os.path.join(path, 'reward_weights.pt'), map_location=self.device)
        self.reward_weights.requires_grad = True
        self.reward_optimizer = torch.optim.Adam([self.reward_weights], lr=1e-4)
        print(f"Model and reward weights loaded from {path}")

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.actor(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state_tensor).item()
        return action.item(), log_prob.item(), value

    def store_transition(self, transition, reward_components):
        self.buffer.append(transition)
        self.buffer_reward_components.append(reward_components)

    def compute_gae(self, rewards, values, dones):
        advantages, gae = [], 0
        values = list(values) + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, epochs=10, batch_size=32):
        states, actions, rewards, dones, log_probs_old, values = zip(*self.buffer)
        reward_components_tensor = torch.tensor(self.buffer_reward_components, dtype=torch.float32).to(self.device)
        softmax_weights = F.softmax(self.reward_weights, dim=0).to(reward_components_tensor.device)
        combined_rewards = (reward_components_tensor @ softmax_weights).detach().cpu().numpy().tolist()

        advantages = self.compute_gae(combined_rewards, values, dones)
        returns = [a + v for a, v in zip(advantages, values)]

        states = torch.FloatTensor(np.array(states)).to(self.device)

        actions = torch.LongTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        #添加标准化处理，避免 advantage 爆炸
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = torch.FloatTensor(returns).to(self.device)

        total_actor_loss, total_critic_loss = 0.0, 0.0
        for _ in range(epochs):
            idx = torch.randperm(states.size(0))
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                b = idx[start:end]
                s_b, a_b, adv_b, ret_b, old_logp_b = states[b], actions[b], advantages[b], returns[b], log_probs_old[b]

                probs, _ = self.actor(s_b)
                dist = torch.distributions.Categorical(probs)
                logp = dist.log_prob(a_b)
                ratio = torch.exp(logp - old_logp_b)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                #critic_loss = F.mse_loss(self.critic(s_b), ret_b)
                critic_loss = F.smooth_l1_loss(self.critic(s_b), ret_b)
                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()

                #梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

                self.optimizer.step()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        # 联合训练 reward_weights：maximize total reward
        self.reward_optimizer.zero_grad()
        softmax_weights = F.softmax(self.reward_weights, dim=0).to(self.device)
        joint_reward = reward_components_tensor @ softmax_weights
        reward_loss = -torch.mean(joint_reward)
        reward_loss.backward()
        self.reward_optimizer.step()

        self.buffer.clear()
        self.buffer_reward_components.clear()
        return total_actor_loss, total_critic_loss, np.mean(combined_rewards)


def train_agent(df_origin_list, anchor_lists, agent, num_epochs=20, stop_thresh=0.01, max_no_change_steps=6):
    for epoch in range(num_epochs):
        print(f"==== Epoch {epoch+1}/{num_epochs} ====")
        all_rewards = []
        for index,(df_origin,anchors) in enumerate(zip(df_origin_list, anchor_lists)):
            print(f"\t\tProcessing {index+1}/{len(df_origin_list)}")
            for anchor in anchors:
                current = anchor.copy()
                last_anchor = current
                no_change_counter = 0
                done = False
                for _ in range(30):
                    state = get_feature(current, df_origin)
                    action, logp, value = agent.select_action(state)
                    action_dict = agent.action_space[action]
                    new_anchor = adjust_anchor_box(df_origin, current, action_dict["type"], action_dict["param"])
                    #print(f"\told anchor:{current}")
                    #print(f"\tnew anchor:{new_anchor}")
                    #print(f"\taction:{action_dict['type']} {action_dict['param']}")

                    weight_list = F.softmax(agent.reward_weights, dim=0).detach().cpu().numpy().tolist()
                    reward, reward_components = calculate_reward_new(current, new_anchor, df_origin, weight_list)
                    agent.store_transition((state, action, reward, False, logp, value), reward_components)

                    all_rewards.append(reward)

                    if np.linalg.norm(np.array(new_anchor) - np.array(last_anchor)) < stop_thresh:
                        no_change_counter += 1
                        if no_change_counter >= max_no_change_steps:
                            break
                    else:
                        no_change_counter = 0

                    last_anchor = new_anchor
                    current = new_anchor

        actor_loss, critic_loss, avg_reward = agent.update()
        print(f"[Epoch {epoch+1}/{num_epochs}] Avg Reward: {avg_reward:.4f} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        softmax_weights = F.softmax(agent.reward_weights, dim=0).detach().cpu().numpy()
        print(f"Reward Weights: {np.round(softmax_weights, 4)}")


def test_agent(df_origin, anchor_list, agent, stop_thresh=0.002, max_no_change_steps=6):
    agent.reward_weights.requires_grad = False
    optimized_anchors = []

    for idx, anchor in enumerate(anchor_list):
        current = anchor.copy()
        last_anchor = current
        no_change_counter = 0
        reward_trace = []

        for _ in range(30):
            state = get_feature(current, df_origin)
            action, _, _ = agent.select_action(state)
            action_dict = agent.action_space[action]
            new_anchor = adjust_anchor_box(df_origin, current, action_dict["type"], action_dict["param"])

            weight_list = F.softmax(agent.reward_weights, dim=0).detach().cpu().numpy().tolist()
            reward, _ = calculate_reward_new(current, new_anchor, df_origin, weight_list)
            reward_trace.append(reward)

            if np.linalg.norm(np.array(new_anchor) - np.array(last_anchor)) < stop_thresh:
                no_change_counter += 1
                if no_change_counter >= max_no_change_steps:
                    break
            else:
                no_change_counter = 0

            last_anchor = new_anchor
            current = new_anchor

        print(f"[TEST] Anchor {idx} | Total Reward: {sum(reward_trace):.4f}")
        optimized_anchors.append(current)

    return optimized_anchors

def process_csv_for_init(file_path):
    '''
    初始化加载数据集，处理单个csv文件
    之所以要抽象出一个处理单个文件的函数来，是因为需要做多进程操作

    为了确保返回的数据对应性正确，关键是：
    每个文件一个任务；
    每个任务的返回值是 (df, anchors, global_features)；
    最后统一收集返回值，用 zip 拆成 df_origin_list、anchor_lists 和 global_feature_lists。
    '''
    # 读取文件并做归一化
    df_origin = pd.read_csv(file_path)
    df = df_origin.iloc[:, 1:]
    df=df_normalization(df)
    # 初始化锚框
    anchors = generate_initial_anchors(df, 100)


    #plot_greyscale_for_singledf_with_anthor(df,anchors,f"images/init_img/init_img{time.time()}.png")
    return df, anchors


DISCRETE_ACTIONS = [
    {"type": "move", "param": "up"},
    {"type": "move", "param": "down"},
    {"type": "move", "param": "left"},
    {"type": "move", "param": "right"},
    {"type": "resize_width", "param": "shrink"},
    {"type": "resize_width", "param": "expand"},
    {"type": "resize_height", "param": "shrink"},
    {"type": "resize_height", "param": "expand"}
]



# -------------------------------
# 主函数：加载数据、训练、测试
# -------------------------------
if __name__ == "__main__":
    print("--------------------初始化智能体----------------------")
    # 状态向量维度为68（前64为区域细节，后4为锚框特征）
    input_dim = 68
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:"+str(device))
    agent = PPOAgent(input_dim=input_dim, action_space=DISCRETE_ACTIONS, device=device)

    #在之前模型基础上继续训练
    agent.load("weights/ppo_anchor_adjust5")
    print("----------------加载训练集--------------------")
    '''
    初始化加载训练集，
    因为每一个频谱文件都要锚框初始化，所以这里干脆采用并行加载的方式
    '''
    dataset_path = "dataset/train"
    csv_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".csv")]

    # 并行处理所有CSV文件
    results = Parallel(n_jobs=8)(delayed(process_csv_for_init)(path) for path in csv_files)

    # 拆分结果
    df_origin_list, anchor_lists = zip(*results)

    # 后续还要用到这两个列表，转换为 list
    df_origin_list = list(df_origin_list)
    anchor_lists = list(anchor_lists)



    print("#####################准备开始训练强化学习的agent##########################")
    # 训练智能体
    train_agent(df_origin_list, anchor_lists,agent=agent,num_epochs=10)
    agent.save("weights/ppo_anchor_adjust6")
    print("训练完成，模型已保存。")

    print("---------------------准备开始测试--------------------------")
    #测试这一部分其实应该放到detect.py里搞得了
    dataset_path = "dataset/train"
    csv_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".csv")]

    # 并行处理所有CSV文件
    results = Parallel(n_jobs=8)(delayed(process_csv_for_init)(path) for path in csv_files)

    # 拆分结果
    df_origin_list, anchor_lists = zip(*results)

    # 后续还要用到这两个列表，转换为 list
    df_origin_list = list(df_origin_list)
    anchor_lists = list(anchor_lists)

    # 加载模型
    agent.load("weights/ppo_anchor_adjust6")

    for index,(df,anchor_list) in enumerate(zip(df_origin_list,anchor_lists)):
        test_df = df

        plot_greyscale_for_singledf_with_anthor_and_score(test_df, anchor_list, f"images/all/{index}模型调整前.png")
        optimized_anchors = test_agent(test_df, anchor_list, agent)
        plot_greyscale_for_singledf_with_anthor_and_score(test_df, optimized_anchors, f"images/all/{index}模型调整后.png")
        merged_anchors=merge_anchor(optimized_anchors, iou_threshold=0.01)
        plot_greyscale_for_singledf_with_anthor_and_score(test_df, merged_anchors,
                                                          f"images/all/{index}合并锚框后.png")