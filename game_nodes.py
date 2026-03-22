# -*- coding: utf-8 -*-
"""
LangGraph 游戏节点定义
街道流程：preflop -> flop -> turn -> river -> showdown
"""

from typing import TypedDict, Annotated, Literal
import operator
from poker_core import create_deck, determine_winner, Card
from agent_config import call_llm_for_action, AGENT_PROFILES
from game_display import (
    print_player_action, print_dealer_info,
    print_winner_info, print_community_cards
)


class GameState(TypedDict):
    hand_number: int
    chips: dict
    pot: int
    community_cards: list
    deck: list
    players_info: dict
    messages: Annotated[list[dict], operator.add]
    current_player: str
    player_order: list
    stage: str
    hand_ended: bool
    winner: str
    street_finished: bool


def _is_active(name, players_info, chips):
    if players_info.get(name, {}).get("folded", False):
        return False
    if chips.get(name, 0) <= 0:
        return False
    return True


def _next_player(current, order):
    return order[(order.index(current) + 1) % len(order)]


def _is_last_player(current, order):
    return order.index(current) == len(order) - 1


def _active_players(state):
    return [n for n in state["player_order"]
            if _is_active(n, state["players_info"], state["chips"])]


def _street_bets_balanced(players_info, chips, active):
    """
    判断本街是否所有需要行动的玩家都已行动且下注平衡。
    规则：
    - 未 fold 的玩家中，chips > 0 的玩家必须已经行动过
    - 所有 chips > 0 的活跃玩家下注额必须等于当前最高下注
      （all-in 玩家允许下注额低于最高下注，视为已无法跟注）
    """
    if not active:
        return True
    # 所有有筹码的活跃玩家都必须已行动
    for n in active:
        if players_info[n].get("action", "") == "":
            return False
    # 计算所有未 fold 玩家（含 all-in）中的最高下注
    all_bets = [players_info[n].get("bet", 0) for n in players_info
                if not players_info[n].get("folded", False)]
    max_bet = max(all_bets) if all_bets else 0
    # 有筹码的活跃玩家下注必须等于最高下注
    for n in active:
        if players_info[n].get("bet", 0) != max_bet:
            return False
    return True


def _reset_street_bets(players_info):
    result = {}
    for name, info in players_info.items():
        result[name] = dict(info)
        result[name]["bet"] = 0
        result[name]["action"] = ""
    return result


def dealer_node(state: GameState) -> dict:
    deck = create_deck()
    order = state["player_order"]
    players_info = {}
    for name in order:
        hole_cards = [deck.pop(), deck.pop()]
        players_info[name] = {
            "hole_cards": hole_cards,
            "bet": 0,
            "total_bet": 0,
            "folded": False,
            "action": "",
            "is_human": state["players_info"].get(name, {}).get("is_human", False),
        }
    print_dealer_info(players_info)
    return {
        "players_info": players_info,
        "deck": deck,
        "community_cards": [],
        "pot": 0,
        "stage": "preflop",
        "current_player": order[0],
        "hand_ended": False,
        "winner": "",
        "street_finished": False,
    }


def deal_community_node(state: GameState) -> dict:
    deck = list(state["deck"])
    community = list(state["community_cards"])
    stage = state["stage"]
    if stage == "preflop":
        community += [deck.pop(), deck.pop(), deck.pop()]
        next_stage = "flop"
    elif stage == "flop":
        community += [deck.pop()]
        next_stage = "turn"
    elif stage == "turn":
        community += [deck.pop()]
        next_stage = "river"
    else:
        next_stage = stage
    print_community_cards(next_stage, community)
    new_players_info = _reset_street_bets(state["players_info"])
    order = state["player_order"]
    first_active = next(
        (n for n in order if _is_active(n, new_players_info, state["chips"])),
        order[0]
    )
    return {
        "deck": deck,
        "community_cards": community,
        "stage": next_stage,
        "players_info": new_players_info,
        "current_player": first_active,
        "street_finished": False,
    }


def player_action_node(state: GameState) -> dict:
    name = state["current_player"]
    order = state["player_order"]
    next_name = _next_player(name, order)
    is_last = _is_last_player(name, order)

    if not _is_active(name, state["players_info"], state["chips"]):
        active = _active_players(state)
        street_finished = _street_bets_balanced(state["players_info"], state["chips"], active)
        return {
            "current_player": next_name,
            "street_finished": street_finished,
            "messages": [],
        }

    hole_cards = state["players_info"][name]["hole_cards"]
    my_chips = state["chips"].get(name, 0)
    current_max_bet = max(p.get("bet", 0) for p in state["players_info"].values())
    my_bet = state["players_info"][name].get("bet", 0)
    to_call = current_max_bet - my_bet

    community = state["community_cards"]
    community_str = " ".join(community) if community else "（尚未翻牌）"
    stage = state["stage"]
    pot = state["pot"]

    ctx_lines = [
        "当前街道：" + stage,
        "公共牌：" + community_str,
        "对手行动：",
    ]
    for other in order:
        if other == name:
            continue
        info = state["players_info"].get(other, {})
        if info.get("action"):
            ctx_lines.append("  " + other + "：" + info["action"] + "（下注" + str(info.get("bet", 0)) + "）")
    ctx_lines.append("")
    ctx_lines.append("当前底池：" + str(pot))
    ctx_lines.append("你当前筹码：" + str(my_chips) + "（不能超过此数下注）")
    ctx_lines.append("你本轮已下注：" + str(my_bet) + "，当前最高下注：" + str(current_max_bet) + "，需跟注金额：" + str(to_call))
    if to_call == 0:
        ctx_lines.append("提示：to_call=0，check 零成本合法，fold 为劣后选择，请优先考虑 check 或 raise。")
    else:
        ctx_lines.append("提示：跟注需 " + str(to_call) + " 筹码，可选 call/raise/fold。")
    game_context = "\n".join(ctx_lines) + "\n"

    is_human = state["players_info"][name].get("is_human", False)
    if is_human:
        action, amount, speech = _human_input(name, hole_cards, game_context, to_call, my_chips)
    else:
        action, amount, speech = call_llm_for_action(name, hole_cards, game_context)

    amount = max(0, min(amount, my_chips))
    if action == "fold":
        amount = 0
    elif action == "check":
        amount = 0
    elif action == "call":
        amount = min(to_call, my_chips)
    elif action == "raise":
        amount = max(0, min(amount, my_chips))
    else:
        action = "check"
        amount = 0

    chips_after_action = my_chips - amount
    print_player_action(name, action, amount, speech , chips_after_action)

    new_players_info = dict(state["players_info"])
    new_players_info[name] = dict(new_players_info[name])
    new_players_info[name]["action"] = action
    new_players_info[name]["bet"] = new_players_info[name].get("bet", 0) + amount
    new_players_info[name]["total_bet"] = new_players_info[name].get("total_bet", 0) + amount
    if action == "fold":
        new_players_info[name]["folded"] = True
    # raise 时，重置其他活跃玩家的 action，让他们重新行动
    if action == "raise":
        for other in order:
            if other == name:
                continue
            if not new_players_info[other].get("folded", False):
                other_info = dict(new_players_info[other])
                other_info["action"] = ""
                new_players_info[other] = other_info

    new_pot = state["pot"] + amount
    new_chips = dict(state["chips"])
    new_chips[name] -= amount

    active = [n for n in order if _is_active(n, new_players_info, new_chips)]
    street_finished = _street_bets_balanced(new_players_info, new_chips, active)

    return {
        "pot": new_pot,
        "chips": new_chips,
        "players_info": new_players_info,
        "current_player": next_name,
        "street_finished": street_finished,
        "messages": [{"agent": name, "action": action, "speech": speech}],
    }


def _human_input(name, hole_cards, game_context, to_call, my_chips):
    cards_str = " ".join([c if isinstance(c, str) else str(c) for c in hole_cards])
    print("\n你的手牌：" + cards_str)
    print(game_context)
    print("可选行动：fold / check / call / raise <金额>")
    raw = input(name + " 请输入行动：").strip()
    parts = raw.split()
    action = parts[0].lower() if parts else "check"
    amount = int(parts[1]) if len(parts) > 1 else 0
    if action not in ["fold", "check", "call", "raise"]:
        action = "check"
    if action in ["fold", "check"]:
        amount = 0
    return (action, amount, "")


def route_player_action(state: GameState) -> str:
    """
    player_action 后的路由：
    - 只剩1人未 fold -> judge（其余人弃牌）
    - 本街结束且是 river -> judge
    - 本街结束且不是 river -> deal_community
    - 本街未结束 -> continue
    """
    # 未 fold 的玩家数
    not_folded = [n for n in state["player_order"]
                  if not state["players_info"].get(n, {}).get("folded", False)]
    if len(not_folded) <= 1:
        return "judge"

    active = _active_players(state)  # 有筹码且未 fold
    # 所有未 fold 的人都 all-in（active 为空），直接发完公共牌到 showdown
    if len(active) == 0:
        if state["stage"] == "river":
            return "judge"
        return "deal_community"

    if state.get("street_finished", False):
        if state["stage"] == "river":
            return "judge"
        return "deal_community"
    return "continue"


def judge_winner_node(state: GameState) -> dict:
    winner = determine_winner(state["players_info"], state["community_cards"])
    new_chips = dict(state["chips"])
    if winner is None:
        print_winner_info("无人获胜", state["pot"], new_chips)
        return {"winner": "", "chips": new_chips, "hand_ended": True}
    new_chips[winner] += state["pot"]
    print_winner_info(winner, state["pot"], new_chips)
    return {"winner": winner, "chips": new_chips, "hand_ended": True}
