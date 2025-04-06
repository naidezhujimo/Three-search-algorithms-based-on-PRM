import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class MiniLM(nn.Module):
    def __init__(self, vocab_size=10, hidden_dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def generate(self, prompt, max_length=1):
        current_seq = prompt.tolist()
        for _ in range(max_length):
            input_tensor = torch.tensor(current_seq).unsqueeze(0)
            with torch.no_grad():
                logits = self.forward(input_tensor)
            next_token = torch.argmax(logits, dim=-1).item()
            current_seq.append(next_token)
        return current_seq[-max_length:]


class PRM(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.scorer(x)


def test_prm_training():
    lm = MiniLM()
    prm = PRM()
    optimizer = optim.Adam(prm.parameters(), lr=0.001)
    
    step_embeddings = torch.randn(100, 8)
    mc_returns = torch.rand(100, 1)
    
    for epoch in range(10):
        pred = prm(step_embeddings)
        loss = nn.MSELoss()(pred, mc_returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


def test_best_of_n(lm, prm):
    prompt = torch.tensor([2, 0, 3])
    candidates = [lm.generate(prompt, max_length=1) for _ in range(5)]
    
    candidate_data = []
    for cand in candidates:
        # 获取生成的token（单个值）
        generated_token = cand[0]  # cand是长度为1的列表
        # 生成随机嵌入（与模型实际使用保持一致）
        step_embs = torch.randn(3, 8)  # 假设3步历史
        # 计算评分（取均值更合理）
        score = prm(step_embs).mean().item()  # 修改为均值
        candidate_data.append((generated_token, score))
    
    # 找出最高分索引（避免浮点精度问题）
    max_index = np.argmax([c[1] for c in candidate_data])
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(
        x=range(len(candidate_data)),
        height=[c[1] for c in candidate_data],
        color=['red' if i == max_index else 'blue' for i in range(len(candidate_data))],
        alpha=0.7
    )
    
    # 设置清晰的标签
    plt.title("Best-of-N Sampling Results", fontsize=12)
    plt.xlabel("Candidate Tokens", fontsize=10)
    plt.ylabel("PRM Score", fontsize=10)
    plt.xticks(
        ticks=range(len(candidate_data)),
        labels=[str(c[0]) for c in candidate_data],  # 直接显示token数值
        rotation=0
    )
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    plt.savefig("best_of_n.png", dpi=150)
    plt.close()
 

def beam_search(lm, beam_width=2):
    prompt = torch.tensor([2, 0, 3])
    sequences = [[list(prompt), 0.0]]
    search_tree = defaultdict(list)
    
    # 搜索过程保持不变
    for step in range(3):
        new_seqs = []
        for seq_idx, (seq, score) in enumerate(sequences):
            logits = lm(torch.tensor(seq).unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_tokens = torch.topk(probs, beam_width)
            
            for p, t in zip(top_probs[0], top_tokens[0]):
                new_seq = seq + [t.item()]
                new_score = score + torch.log(p).item()
                new_seqs.append((new_seq, new_score))
                search_tree[step].append((new_seq, new_score))
        
        sequences = sorted(new_seqs, key=lambda x: -x[1])[:beam_width]
    

    plt.figure(figsize=(12, 8))
    G = nx.DiGraph()
    pos = {}
    
    # 计算节点位置
    level_spacing = 3.0
    node_spacing = 1.5
    for level in sorted(search_tree.keys()):
        nodes = search_tree[level]
        y_base = -level * level_spacing
        for idx, (seq, score) in enumerate(nodes):
            # 水平均匀分布节点
            x = (idx - len(nodes)/2) * node_spacing
            pos[(level, tuple(seq))] = (x, y_base)
            
            # 添加节点和边
            parent_seq = seq[:-1]
            if level > 0:
                G.add_edge((level-1, tuple(parent_seq)), (level, tuple(seq)))
    
    # 标记最终路径
    final_paths = [seq for seq, _ in sequences]
    node_colors = []
    edge_colors = []
    for node in G.nodes():
        seq = node[1]
        is_in_final = any(seq == tuple(p[:len(seq)]) for p in final_paths)
        node_colors.append('limegreen' if is_in_final else 'lightgray')
        
        # 标记最终路径边
        if node[0] > 0:
            edge = (list(G.predecessors(node))[0], node)
            edge_colors.append('red' if is_in_final else 'gray')
    
    # 绘制节点和边
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors,
        node_size=2500,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5
    )
    
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        arrowsize=20,
        width=2 if edge_colors == 'red' else 1,
        style='dashed' if edge_colors == 'gray' else 'solid'
    )
    
    # 添加标签（显示token和累积概率）
    labels = {
        node: f"Token: {node[1][-1]}\nScore: {score:.2f}"
        for level in search_tree
        for (node, (seq, score)) in [( (level, tuple(seq)), (seq, score)) 
                                    for seq, score in search_tree[level]]
    }
    nx.draw_networkx_labels(
        G, pos, 
        labels=labels,
        font_size=10,
        verticalalignment='center'
    )
    
    # 添加图例和装饰
    plt.title("Beam Search Visualization (Width=2)", fontsize=14)
    plt.gca().invert_yaxis()  # 让根节点显示在顶部
    plt.axis('off')
    
    # 手动添加图例
    plt.scatter([], [], c='limegreen', s=200, label='Final Path')
    plt.scatter([], [], c='lightgray', s=200, label='Pruned Path')
    plt.plot([], [], color='red', linestyle='-', label='Final Edges')
    plt.plot([], [], color='gray', linestyle='--', label='Pruned Edges')
    plt.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig("beam_search_tree.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Final sequence: {sequences[0][0]}")
    return sequences[0][0]

def lookahead_search(lm, prm, horizon=2):
    current_seq = [2, 0, 3]
    decision_points = []
    
    for step in range(3):
        logits = lm(torch.tensor(current_seq).unsqueeze(0))
        top_tokens = torch.topk(logits, 2)[1][0]  # 取top-2候选
        candidates = []
        
        # 对每个候选token进行前瞻
        for token in top_tokens:
            temp_seq = current_seq.copy()
            temp_seq.append(token.item())
            future_score = 0.0
            
            # 模拟未来horizon步
            for _ in range(horizon):
                next_logits = lm(torch.tensor(temp_seq).unsqueeze(0))
                next_token = torch.argmax(next_logits).item()
                temp_seq.append(next_token)
                # 使用真实模型嵌入计算奖励
                with torch.no_grad():
                    emb = lm.embed(torch.tensor([[next_token]]))
                    future_score += prm(emb.mean(dim=1)).item()
            
            # 综合评分 = 当前token概率 + 未来奖励
            current_prob = torch.softmax(logits, dim=-1)[0, token].item()
            total_score = np.log(current_prob) + future_score
            candidates.append((token.item(), total_score))
        
        # 记录决策点
        decision_points.append({
            "step": step + 1,
            "candidates": candidates,
            "chosen": max(candidates, key=lambda x: x[1])
        })
        current_seq.append(max(candidates, key=lambda x: x[1])[0])
    

    plt.figure(figsize=(10, 6))
    
    # 绘制候选和选择点
    for i, dp in enumerate(decision_points):
        x_pos = [i+1] * len(dp["candidates"])
        scores = [c[1] for c in dp["candidates"]]
        tokens = [str(c[0]) for c in dp["candidates"]]
        
        # 绘制候选点（带数值标签）
        scatter = plt.scatter(
            x_pos, scores,
            c='blue', s=100, alpha=0.7,
            label='Candidates' if i==0 else ""
        )
        
        # 添加token数值标签
        for (x, y, t) in zip(x_pos, scores, tokens):
            plt.text(x+0.05, y, f"Token {t}\n{y:.2f}",
                    ha='left', va='center', fontsize=8)
        
        # 绘制选择点
        chosen_x = i+1
        chosen_y = dp["chosen"][1]
        plt.scatter(
            chosen_x, chosen_y,
            c='red', marker='*', s=400,
            label='Chosen' if i==0 else "",
            edgecolors='black', linewidths=0.5
        )
    
    # 连接选择点形成路径
    path_x = [i+1 for i in range(len(decision_points))]
    path_y = [dp["chosen"][1] for dp in decision_points]
    plt.plot(path_x, path_y, 'r--', alpha=0.5, label='Selection Path')
    
    # 图表装饰
    plt.title("Lookahead Search Decision Process (Horizon=2)", fontsize=14)
    plt.xlabel("Decision Step", fontsize=12)
    plt.ylabel("Total Score (logP + Future Reward)", fontsize=12)
    plt.xticks(
        ticks=[1,2,3],
        labels=[f"Step {i}\nCurrent token: {decision_points[i-1]['chosen'][0]}" 
               for i in [1,2,3]]
    )
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("lookahead_decision.png", dpi=150)
    plt.close()
    
    print(f"Final sequence: {current_seq}")
    return current_seq

if __name__ == "__main__":
    lm = MiniLM()
    prm = PRM()
    
    print("==== Testing PRM Training ====")
    test_prm_training()
    
    print("\n==== Testing Best-of-N ====")
    test_best_of_n(lm, prm)
    
    print("\n==== Testing Beam Search ====")
    beam_search(lm)
    
    print("\n==== Testing Lookahead Search ====")
    lookahead_search(lm, prm)