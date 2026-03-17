# -*- coding: utf-8 -*-
"""
德州扑克基础模块 - 牌类、发牌、手牌评估
"""

import random


class Card:
    """扑克牌类"""
    SUITS = ["♠", "♥", "♦", "♣"]
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    RANK_VALUES = {r: i for i, r in enumerate(RANKS)}

    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return self.rank + self.suit

    def __repr__(self):
        return str(self)


def create_deck():
    """创建一副完整的扑克牌"""
    deck = []
    for suit in Card.SUITS:
        for rank in Card.RANKS:
            deck.append(Card(suit, rank))
    random.shuffle(deck)
    return deck


def evaluate_hand(hole_cards: list, community_cards: list) -> tuple:
    """
    评估手牌强度（简化版）
    返回 (强度分数, 手牌描述)
    """
    all_cards = hole_cards + community_cards
    ranks = [Card.RANK_VALUES[c.rank] for c in all_cards]
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    
    pairs = sum(1 for count in rank_counts.values() if count >= 2)
    trips = sum(1 for count in rank_counts.values() if count >= 3)
    quads = sum(1 for count in rank_counts.values() if count >= 4)
    
    score = quads * 10000 + trips * 1000 + pairs * 100 + max(ranks)
    
    if quads > 0:
        desc = "四条"
    elif trips > 0:
        desc = "三条"
    elif pairs >= 2:
        desc = "两对"
    elif pairs == 1:
        desc = "一对"
    else:
        desc = "高牌"
    
    return (score, desc)


def determine_winner(players_info: dict, community_cards: list) -> str:
    """判断赢家"""
    best_score = -1
    winner = None
    
    for name, info in players_info.items():
        if info.get("folded", False):
            continue
        score, _ = evaluate_hand(info["hole_cards"], community_cards)
        if score > best_score:
            best_score = score
            winner = name
    
    return winner
