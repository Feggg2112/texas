"""
utils/poker_utils.py - 德州扑克纯工具函数

【本文件职责】
这是「工具层」，不依赖 LangGraph，不调用 LLM，只做数学计算。
所有函数都是无副作用的纯函数，便于独立单元测试。

【treys 库】
用完美哈希表预计算了全部 7462 种牌型分数，查表速度微秒级。
牌字符串格式：Rank+Suit，例：'Ah'=红心A, 'Kd'=方块K, 'Tc'=梅花10
  Rank: 2-9 T J Q K A
  Suit: c(梅花) d(方块) h(红心) s(黑桃)
"""
from __future__ import annotations
import random
from treys import Card, Evaluator

# 模块级单例：只初始化一次，避免每次调用都重新加载哈希表
_evaluator = Evaluator()

# ── 牌面常量 ──────────────────────────────────────────────────────────────────

RANKS = "23456789TJQKA"  # 13种点数，从小到大
SUITS = "cdhs"           # 4种花色

# 列表推导式生成完整52张牌，例：['2c','2d','2h','2s','3c',...,'As']
FULL_DECK: list[str] = [r + s for r in RANKS for s in SUITS]


# ── 发牌函数 ──────────────────────────────────────────────────────────────────

def make_deck() -> list[str]:
    """
    创建并洗好一副牌。

    .copy() 防止修改全局常量 FULL_DECK。
    random.shuffle 原地洗牌（in-place），每局开始调用一次。
    返回值存入 GameState.deck。
    """
    deck = FULL_DECK.copy()
    random.shuffle(deck)
    return deck


def deal_hole_cards(deck: list[str], n_players: int) -> tuple[list[list[str]], list[str]]:
    """
    给 n_players 每人发 2 张底牌。

    【函数式风格】返回 (hole_cards_list, remaining_deck) 而不修改传入的 deck。
    调用方把新 deck 存回 State，保证节点函数无副作用。
    deck.pop() 从末尾取牌，O(1) 操作（比头部删除快）。
    """
    hole_cards: list[list[str]] = []
    for _ in range(n_players):
        hand = [deck.pop(), deck.pop()]
        hole_cards.append(hand)
    return hole_cards, deck


def deal_flop(deck: list[str]) -> tuple[list[str], list[str]]:
    """
    发翻牌（Flop）：先烧1张牌，再发3张公共牌。

    烧牌（burn card）是真实规则，防止有人看到牌背记号作弊。
    翻牌是德扑第一轮公共牌，发完后共3张公共牌在桌上。
    """
    deck.pop()  # 烧牌，丢弃不用
    flop = [deck.pop(), deck.pop(), deck.pop()]
    return flop, deck


def deal_one(deck: list[str]) -> tuple[str, list[str]]:
    """
    发1张公共牌（转牌 Turn 或河牌 River），先烧1张。

    转牌后桌面共4张公共牌，河牌后共5张。
    5张公共牌 + 2张底牌 = 7张中取最优5张组合，这就是 treys 做的事。
    """
    deck.pop()  # 烧牌
    card = deck.pop()
    return card, deck


# ── 手牌强度评估 ──────────────────────────────────────────────────────────────

def evaluate_hand(hole_cards: list[str], community_cards: list[str]) -> dict:
    """
    用 treys 库评估当前手牌强度（需要至少3张公共牌）。

    【treys 评分说明】
    score：1（皇家同花顺，最强）~ 7462（最差高牌，最弱）
    rank_class：1=皇家同花顺 2=同花顺 3=四条 4=葫芦
                5=同花 6=顺子 7=三条 8=两对 9=一对 10=高牌

    【归一化 strength】
    strength = 1 - (score-1)/7461，线性映射到 [0,1]，1=最强，0=最弱。
    给 LLM 一个直觉数字：「你的手牌强度是 82%」比原始分数 1300 更易理解。

    【try/except 容错】
    公共牌不足3张时 treys 会抛异常，捕获后返回最弱默认值，不中断图执行。
    """
    try:
        # Card.new() 把字符串转成 treys 内部整数位图（bitmask）
        h = [Card.new(c) for c in hole_cards]
        b = [Card.new(c) for c in community_cards]
        # 注意参数顺序：board（公共牌）在前，hand（底牌）在后
        score = _evaluator.evaluate(b, h)
        rank_class = _evaluator.get_rank_class(score)
        rank_string = _evaluator.class_to_string(rank_class)  # 转为可读牌型名
        # treys 分数越小越强，逆向归一化让「数值大=牌强」更直觉
        strength = 1.0 - (score - 1) / 7461.0
        return {
            "score": score,
            "rank_class": rank_class,
            "rank_string": rank_string,
            "strength": round(strength, 4),
        }
    except Exception:
        # 公共牌不足 3 张时无法评估，返回最弱默认值
        return {"score": 7462, "rank_class": 9, "rank_string": "High Card", "strength": 0.0}


def hand_strength_preflop(hole_cards: list[str]) -> float:
    """
    翻前手牌强度快速估算（Chen 公式简化版），返回 0~1。

    【为什么翻前不用 treys？】
    翻前只有2张手牌，没有公共牌，treys 需要至少5张牌才能评估。
    这里用 Chen 公式简化版：高牌价值 + 对子加成 + 同花加成 + 连张加成。

    【更精确的替代方案（后续优化）】
    蒙特卡洛模拟：随机发5张公共牌，跑1000次，统计胜率。精确但慢。
    """
    if len(hole_cards) < 2:
        return 0.0
    r1, r2 = hole_cards[0][0], hole_cards[1][0]
    suited = hole_cards[0][1] == hole_cards[1][1]  # 是否同花

    # 建立点数到整数值的映射：2=2, 3=3, ..., T=10, J=11, Q=12, K=13, A=14
    rank_val = {r: i for i, r in enumerate(RANKS, 2)}
    # v1 >= v2，高牌排前面
    v1, v2 = sorted([rank_val.get(r1, 2), rank_val.get(r2, 2)], reverse=True)

    score = v1 * 0.5   # 基础分：高牌价值越大越好
    if v1 == v2:        # 口袋对子（如 AA, KK）大幅加分
        score = max(score * 2, 5)
    if suited:          # 同花：增加同花顺听牌可能
        score += 2
    gap = v1 - v2       # 两张牌点数差距，差距越小连张价值越高
    if gap == 1:
        score += 1      # 连张（如 JT），加1分
    elif gap <= 3:
        score -= gap - 1  # 间隔越大扣分越多

    return min(max(score / 20.0, 0.0), 1.0)  # 归一化到 [0, 1]


# ── 底池赔率计算 ──────────────────────────────────────────────────────────────

def calc_pot_odds(call_amount: int, pot: int) -> float:
    """
    计算底池赔率（Pot Odds）。

    【公式】pot_odds = call_amount / (pot + call_amount)

    这代表「这次跟注，你投入占总底池的比例」。
    如果你的胜率 > pot_odds，跟注期望值为正（合算）。

    例：底池100，需跟注25 → pot_odds = 25/125 = 20%
    若判断胜率 > 20%，则跟注有正期望值，应该跟注。
    """
    if call_amount <= 0:
        return 0.0  # 无需跟注（可过牌），赔率为 0
    total = pot + call_amount
    return round(call_amount / total, 4) if total > 0 else 0.0


def calc_equity_needed(pot_odds: float) -> float:
    """
    跟注所需的最低胜率（保证期望值 >= 0）。

    【推导】EV = win_rate * pot - (1-win_rate) * call_amount >= 0
    解出 win_rate >= call_amount / (pot + call_amount) = pot_odds
    所以最低胜率在数值上就等于 pot_odds。
    这个函数作为语义更清晰的别名存在。
    """
    return pot_odds


# ── 摊牌结算 ──────────────────────────────────────────────────────────────────

def determine_winners(
    active_players: list[dict],  # 每个元素含 'id', 'hole_cards'
    community_cards: list[str],
) -> list[int]:
    """
    比较所有活跃玩家手牌，返回赢家 id 列表（平局时多人）。

    【算法】遍历所有活跃玩家，用 treys 评估每人手牌分数，
    保留分数最低（最强）的玩家 id。分数相同则平局，两人都入赢家列表。

    active_players 格式：[{'id': int, 'hole_cards': ['Ah', 'Kd']}, ...]
    """
    best_score = 9999
    winners: list[int] = []

    for p in active_players:
        result = evaluate_hand(p["hole_cards"], community_cards)
        score = result["score"]
        if score < best_score:    # 找到更强的牌，更新最佳
            best_score = score
            winners = [p["id"]]
        elif score == best_score: # 平局，加入赢家列表
            winners.append(p["id"])

    return winners


# ── 格式化工具 ────────────────────────────────────────────────────────────────

def cards_to_str(cards: list[str]) -> str:
    """将牌列表转为可读字符串，如 ['Ah', 'Kd'] -> 'Ah Kd'。"""
    return " ".join(cards) if cards else "(无)"


def format_action_history(history: list[dict]) -> str:
    """
    将行动历史格式化为自然语言，供 LLM 使用。
    """
    if not history:
        return "（暂无行动记录）"
    lines = []
    for entry in history:
        act = entry.get("action", "?")
        name = entry.get("player_name", entry.get("player_id", "?"))
        amount = entry.get("amount", 0)
        street = entry.get("street", "")
        if act == "showdown":
            winners = entry.get("winners", [])
            pot = entry.get("pot", 0)
            lines.append(f"[{street}] 摊牌 - 赢家: {', '.join(winners)}，底池: {pot}")
        elif act in ("small_blind", "big_blind"):
            lines.append(f"[{street}] {name}: {act} {amount}")
        elif amount:
            lines.append(f"[{street}] {name}: {act} {amount}")
        else:
            lines.append(f"[{street}] {name}: {act}")
    return "\n".join(lines)

    