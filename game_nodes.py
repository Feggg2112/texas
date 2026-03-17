# -*- coding: utf-8 -*-
"""
LangGraph 游戏节点定义
"""

from typing import TypedDict, Annotated, Literal
import operator
from poker_core import create_deck, determine_winner
from agent_config import call_llm_for_action, AGENT_PROFILES
from game_display import print_player_action, print_dealer_info, print_winner_info


class GameState(TypedDict):
    """游戏状态"""
    hand_number: int
    chips: dict
    pot: int
    community_cards: list
    players_info: dict
    messages: Annotated[list[dict], operator.add]
    current_player: str
    stage: str
    hand_ended: bool
    winner: str
    round_count: int


# ══════════════════════════════════════════════════════════════
# 发牌节点
# ══════════════════════════════════════════════════════════════

def dealer_node(state: GameState) -> dict:
    """发牌节点"""
    deck = create_deck()
    
    players_info = {}
    for i, name in enumerate(["老鹰", "小辣椒", "老钱"]):
        hole_cards = [deck.pop(), deck.pop()]
        players_info[name] = {
            "hole_cards": hole_cards,
            "bet": 0,
            "total_bet": 0,
            "folded": False,
            "action": "",
        }
    
    print_dealer_info(players_info)
    
    return {
        "players_info": players_info,
        "community_cards": [],
        "pot": 0,
        "stage": "preflop",
        "current_player": "老鹰",
        "hand_ended": False,
        "winner": "",
        "round_count": 1,
    }


# ══════════════════════════════════════════════════════════════
# 三个玩家的行动节点
# ══════════════════════════════════════════════════════════════

def agent_laoying_action(state: GameState) -> dict:
    """老鹰行动"""
    name = "老鹰"
    if state["players_info"][name].get("folded", False):
        return {
            "current_player": "小辣椒",
            "messages": [],
        }
    
    hole_cards = state["players_info"][name]["hole_cards"]
    
    game_context = "对手行动：\n"
    for other_name in ["小辣椒", "老钱"]:
        if other_name in state["players_info"]:
            other_info = state["players_info"][other_name]
            if other_info.get("action"):
                game_context += f"  {other_name}：{other_info['action']}\n"
    
    action, amount, speech = call_llm_for_action(name, hole_cards, game_context)
    print_player_action(name, action, amount, speech)
    
    state["players_info"][name]["action"] = action
    state["players_info"][name]["bet"] = amount
    state["players_info"][name]["total_bet"] = state["players_info"][name].get("total_bet", 0) + amount
    if action == "fold":
        state["players_info"][name]["folded"] = True
    
    new_pot = state["pot"] + amount
    
    return {
        "pot": new_pot,
        "current_player": "小辣椒",
        "messages": [{"agent": name, "action": action, "speech": speech}],
    }


def agent_xiaolajiao_action(state: GameState) -> dict:
    """小辣椒行动"""
    name = "小辣椒"
    if state["players_info"][name].get("folded", False):
        return {
            "current_player": "老钱",
            "messages": [],
        }
    
    hole_cards = state["players_info"][name]["hole_cards"]
    
    game_context = "对手行动：\n"
    for other_name in ["老鹰", "老钱"]:
        if other_name in state["players_info"]:
            other_info = state["players_info"][other_name]
            if other_info.get("action"):
                game_context += f"  {other_name}：{other_info['action']}\n"
    
    action, amount, speech = call_llm_for_action(name, hole_cards, game_context)
    print_player_action(name, action, amount, speech)
    
    state["players_info"][name]["action"] = action
    state["players_info"][name]["bet"] = amount
    state["players_info"][name]["total_bet"] = state["players_info"][name].get("total_bet", 0) + amount
    if action == "fold":
        state["players_info"][name]["folded"] = True
    
    new_pot = state["pot"] + amount
    
    return {
        "pot": new_pot,
        "current_player": "老钱",
        "messages": [{"agent": name, "action": action, "speech": speech}],
    }


def agent_laoqian_action(state: GameState) -> dict:
    """老钱行动"""
    name = "老钱"
    if state["players_info"][name].get("folded", False):
        active_players = sum(1 for p in state["players_info"].values() if not p.get("folded", False))
        hand_ended = active_players <= 1
        return {
            "current_player": "",
            "hand_ended": hand_ended,
            "messages": [],
        }
    
    hole_cards = state["players_info"][name]["hole_cards"]
    
    game_context = "对手行动：\n"
    for other_name in ["老鹰", "小辣椒"]:
        if other_name in state["players_info"]:
            other_info = state["players_info"][other_name]
            if other_info.get("action"):
                game_context += f"  {other_name}：{other_info['action']}\n"
    
    action, amount, speech = call_llm_for_action(name, hole_cards, game_context)
    print_player_action(name, action, amount, speech)
    
    state["players_info"][name]["action"] = action
    state["players_info"][name]["bet"] = amount
    state["players_info"][name]["total_bet"] = state["players_info"][name].get("total_bet", 0) + amount
    if action == "fold":
        state["players_info"][name]["folded"] = True
    
    new_pot = state["pot"] + amount
    
    # 判断是否结束：有人弃牌 或 所有活跃玩家都过牌 或 轮数超过3轮
    active_players = sum(1 for p in state["players_info"].values() if not p.get("folded", False))
    all_checked = all(p.get("action") == "check" for p in state["players_info"].values() if not p.get("folded", False))
    hand_ended = active_players <= 1 or all_checked or state["round_count"] >= 3
    
    new_round = state["round_count"] + 1
    
    return {
        "pot": new_pot,
        "current_player": "",
        "hand_ended": hand_ended,
        "round_count": new_round,
        "messages": [{"agent": name, "action": action, "speech": speech}],
    }


# ══════════════════════════════════════════════════════════════
# 判断赢家节点
# ══════════════════════════════════════════════════════════════

def judge_winner_node(state: GameState) -> dict:
    """判断赢家并结算筹码"""
    winner = determine_winner(state["players_info"], state["community_cards"])
    
    new_chips = dict(state["chips"])
    for name in ["老鹰", "小辣椒", "老钱"]:
        # 使用 total_bet（累积下注）而不是 bet（当前轮下注）
        total_bet = state["players_info"][name].get("total_bet", 0)
        new_chips[name] -= total_bet
    
    new_chips[winner] += state["pot"]
    
    print_winner_info(winner, state["pot"], new_chips)
    
    return {
        "winner": winner,
        "chips": new_chips,
        "hand_ended": True,
    }


# ══════════════════════════════════════════════════════════════
# 条件边函数
# ══════════════════════════════════════════════════════════════

def check_hand_end(state: GameState) -> Literal["judge", "continue"]:
    """判断是否结束本手牌"""
    if state.get("hand_ended", False):
        return "judge"
    return "continue"
