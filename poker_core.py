# -*- coding: utf-8 -*-
"""
德州扑克基础模块 - 牌类、发牌、手牌评估（完整版）
"""

import random
from itertools import combinations


class Card:
    """扑克牌类"""
    SUITS = ["\u2660", "\u2665", "\u2666", "\u2663"]
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    RANK_VALUES = {r: i for i, r in enumerate(RANKS)}

    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return self.rank + self.suit

    def __repr__(self):
        return str(self)

    def to_str(self) -> str:
        """序列化为字符串，用于存入 GameState"""
        return self.rank + self.suit

    @classmethod
    def from_str(cls, s: str) -> "Card":
        """从字符串反序列化"""
        return cls(suit=s[1], rank=s[0])


def create_deck() -> list:
    """创建一副完整的扑克牌，返回字符串列表（便于 LangGraph State 序列化）"""
    deck = []
    for suit in Card.SUITS:
        for rank in Card.RANKS:
            deck.append(Card(suit, rank).to_str())
    random.shuffle(deck)
    return deck


def _cards_from_strs(card_strs: list) -> list:
    """将字符串列表转为 Card 对象列表"""
    return [Card.from_str(s) if isinstance(s, str) else s for s in card_strs]


# ══════════════════════════════════════════════════════════════
# 完整手牌评估（支持 5~7 张牌，7选5最优）
# ══════════════════════════════════════════════════════════════

# 牌型等级（越大越强）
HAND_RANKS = {
    "高牌":     0,
    "一对":     1,
    "两对":     2,
    "三条":     3,
    "顺子":     4,
    "同花":     5,
    "葫芦":     6,
    "四条":     7,
    "同花顺":   8,
    "皇家同花顺": 9,
}


def _evaluate_five(cards: list) -> tuple:
    """
    评估恰好 5 张牌的强度。
    返回 (score_tuple, desc)，score_tuple 可直接比较大小。
    score_tuple 格式：(hand_rank, [决定胜负的点数列表，从高到低])
    """
    ranks = sorted([Card.RANK_VALUES[c.rank] for c in cards], reverse=True)
    suits = [c.suit for c in cards]

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    counts = sorted(rank_counts.values(), reverse=True)   # e.g. [2,2,1] 两对
    count_ranks = sorted(rank_counts.keys(),
                         key=lambda r: (rank_counts[r], r), reverse=True)

    is_flush = len(set(suits)) == 1

    # 顺子检测（含 A-2-3-4-5 低顺）
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks == [12, 3, 2, 1, 0]:  # A-2-3-4-5
            is_straight = True
            straight_high = 3  # 5 high

    # 判断牌型
    if is_straight and is_flush:
        desc = "皇家同花顺" if straight_high == 12 else "同花顺"
        return (HAND_RANKS[desc], [straight_high]), desc

    if counts[0] == 4:
        kicker = [r for r in count_ranks if rank_counts[r] == 1]
        return (HAND_RANKS["四条"], [count_ranks[0]] + kicker), "四条"

    if counts[0] == 3 and counts[1] == 2:
        trip_r = count_ranks[0]
        pair_r = count_ranks[1]
        return (HAND_RANKS["葫芦"], [trip_r, pair_r]), "葫芦"

    if is_flush:
        return (HAND_RANKS["同花"], ranks), "同花"

    if is_straight:
        return (HAND_RANKS["顺子"], [straight_high]), "顺子"

    if counts[0] == 3:
        trip_r = count_ranks[0]
        kickers = sorted([r for r in ranks if rank_counts[r] == 1], reverse=True)
        return (HAND_RANKS["三条"], [trip_r] + kickers), "三条"

    if counts[0] == 2 and counts[1] == 2:
        pair_rs = sorted([r for r in count_ranks if rank_counts[r] == 2], reverse=True)
        kicker = [r for r in count_ranks if rank_counts[r] == 1]
        return (HAND_RANKS["两对"], pair_rs + kicker), "两对"

    if counts[0] == 2:
        pair_r = count_ranks[0]
        kickers = sorted([r for r in ranks if rank_counts[r] == 1], reverse=True)
        return (HAND_RANKS["一对"], [pair_r] + kickers), "一对"

    return (HAND_RANKS["高牌"], ranks), "高牌"


def evaluate_hand(hole_cards: list, community_cards: list) -> tuple:
    """
    评估手牌强度（支持 2~7 张牌，自动 7选5 取最优）。
    hole_cards / community_cards 可以是 Card 对象或字符串。
    返回 (score_tuple, desc)。
    """
    all_cards = _cards_from_strs(hole_cards) + _cards_from_strs(community_cards)

    if len(all_cards) < 5:
        # 公共牌不足5张时，用现有牌评估（preflop/flop 阶段仅供参考）
        padded = all_cards  # 不足时直接评估现有
        score, desc = _evaluate_five(padded) if len(padded) == 5 else _simple_fallback(all_cards)
        return score, desc

    # 7选5：从所有牌中选最优5张
    best_score = None
    best_desc = ""
    for five in combinations(all_cards, 5):
        score, desc = _evaluate_five(list(five))
        if best_score is None or score > best_score:
            best_score = score
            best_desc = desc

    return best_score, best_desc


def _simple_fallback(cards: list) -> tuple:
    """牌数不足5张时的简化评估（preflop阶段）"""
    ranks = sorted([Card.RANK_VALUES[c.rank] for c in cards], reverse=True)
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    counts = sorted(rank_counts.values(), reverse=True)
    count_ranks = sorted(rank_counts.keys(),
                         key=lambda r: (rank_counts[r], r), reverse=True)
    if counts[0] == 2:
        return (HAND_RANKS["一对"], count_ranks), "一对"
    return (HAND_RANKS["高牌"], ranks), "高牌"


def determine_winner(players_info: dict, community_cards: list) -> str:
    """判断赢家，返回赢家名字，全员弃牌返回 None"""
    best_score = None
    winner = None

    for name, info in players_info.items():
        if info.get("folded", False):
            continue
        score, _ = evaluate_hand(info["hole_cards"], community_cards)
        if best_score is None or score > best_score:
            best_score = score
            winner = name

    return winner
