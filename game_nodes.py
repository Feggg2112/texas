# -*- coding: utf-8 -*-
"""
LangGraph 游戏节点定义
街道流程：preflop -> flop -> turn -> river -> showdown
"""

from typing import TypedDict, Annotated
import operator
from poker_core import create_deck, determine_winner
from agent_config import call_llm_for_action
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
    ante: int
    min_open_raise: int
    max_raises_per_street: int
    raises_in_street: int
    last_raise_size: int


def _is_active(name, players_info, chips):
    if players_info.get(name, {}).get("folded", False):
        return False
    if chips.get(name, 0) <= 0:
        return False
    return True


def _next_player(current, order):
    return order[(order.index(current) + 1) % len(order)]


def _active_players(state):
    return [n for n in state["player_order"]
            if _is_active(n, state["players_info"], state["chips"])]


def _alive_players(state):
    return [n for n in state["player_order"] if state["chips"].get(n, 0) > 0]


def _street_bets_balanced(players_info, active):
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
    ante = state.get("ante", 20)
    players_info = {}

    new_chips = dict(state["chips"])
    ante_total = 0

    for name in order:
        is_eliminated = new_chips.get(name, 0) <= 0

        if not is_eliminated:
            ante_paid = min(ante, new_chips[name])
            new_chips[name] -= ante_paid
            ante_total += ante_paid
            is_eliminated = new_chips.get(name, 0) <= 0
        else:
            ante_paid = 0

        hole_cards = [deck.pop(), deck.pop()] if not is_eliminated else []
        players_info[name] = {
            "hole_cards": hole_cards,
            "bet": ante_paid,
            "total_bet": ante_paid,
            "folded": is_eliminated,
            "action": "",
            "is_human": state["players_info"].get(name, {}).get("is_human", False),
        }

    print_dealer_info(players_info)
    if ante_total > 0:
        print("  （前注已收取：每位存活玩家 " + str(ante) + "，当前底池已含前注 " + str(ante_total) + "）")

    return {
        "players_info": players_info,
        "chips": new_chips,
        "deck": deck,
        "community_cards": [],
        "pot": ante_total,
        "stage": "preflop",
        "current_player": order[0],
        "hand_ended": False,
        "winner": "",
        "street_finished": False,
        "raises_in_street": 0,
        "last_raise_size": state.get("min_open_raise", 100),
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
        None
    )
    # 若所有未 fold 玩家都 all-in，打印提示，跳过下注直接发下一张牌
    not_folded = [n for n in order if not new_players_info[n].get("folded", False)]
    active_count = sum(1 for n in not_folded if state["chips"].get(n, 0) > 0)
    if active_count == 0:
        print("  （所有未弃牌玩家均已全押，本轮无需下注）")
    return {
        "deck": deck,
        "community_cards": community,
        "stage": next_stage,
        "players_info": new_players_info,
        "current_player": first_active if first_active else order[0],
        "street_finished": False,
        "raises_in_street": 0,
        "last_raise_size": state.get("min_open_raise", 100),
    }


def player_action_node(state: GameState) -> dict:
    name = state["current_player"]
    order = state["player_order"]
    next_name = _next_player(name, order)

    if not _is_active(name, state["players_info"], state["chips"]):
        active = _active_players(state)
        street_finished = _street_bets_balanced(state["players_info"], active)
        return {
            "current_player": next_name,
            "street_finished": street_finished,
            "messages": [],
        }

    hole_cards = state["players_info"][name]["hole_cards"]
    my_chips = state["chips"].get(name, 0)
    current_max_bet = max(p.get("bet", 0) for p in state["players_info"].values())
    my_bet = state["players_info"][name].get("bet", 0)
    to_call = max(0, current_max_bet - my_bet)

    stage = state["stage"]
    pot = state["pot"]
    raises_in_street = state.get("raises_in_street", 0)
    max_raises_per_street = state.get("max_raises_per_street", 3)
    raise_cap_reached = raises_in_street >= max_raises_per_street
    min_open_raise = state.get("min_open_raise", 100)
    last_raise_size = state.get("last_raise_size", min_open_raise)
    min_raise_size = min_open_raise if current_max_bet == 0 else max(last_raise_size, min_open_raise)
    min_raise_to = current_max_bet + min_raise_size

    community = state["community_cards"]
    community_str = " ".join(community) if community else "（尚未翻牌）"

    ctx_lines = [
        "当前第 " + str(state.get("hand_number", 1)) + " 手（当前为锦标赛模式：坚持到最后一人存活）",
        "当前仍存活玩家数：" + str(len(_alive_players(state))),
        "当前街道：" + stage,
        "公共牌：" + community_str,
        "规则：前注=" + str(state.get("ante", 20)) + "；每街最多加注 " + str(max_raises_per_street) + " 次；本街已加注 " + str(raises_in_street) + " 次",
        "规则：本次最小加注到 " + str(min_raise_to) + "（最小加注增量 " + str(min_raise_size) + "）",
        "对手行动：",
    ]
    for other in order:
        if other == name:
            continue
        info = state["players_info"].get(other, {})
        if info.get("action"):
            ctx_lines.append("  " + other + "：" + info["action"] + "（下注" + str(info.get("bet", 0)) + "）")

    ctx_lines.append("对手剩余筹码：")
    for other in order:
        if other == name:
            continue
        ctx_lines.append("  " + other + "：" + str(state["chips"].get(other, 0)))
    ctx_lines.append("")
    ctx_lines.append("当前底池：" + str(pot))
    ctx_lines.append("你当前筹码：" + str(my_chips) + "（不能超过此数下注）")
    ctx_lines.append("你本轮已下注：" + str(my_bet) + "，当前最高下注：" + str(current_max_bet) + "，需跟注金额：" + str(to_call))
    if raise_cap_reached:
        ctx_lines.append("提示：本街加注次数已到上限，只能 call/check/fold，不能 raise。")
    else:
        ctx_lines.append("提示：若选择 raise，至少加注到 " + str(min_raise_to) + "。")
    if to_call == 0:
        ctx_lines.append("提示：to_call=0，check 零成本合法。")
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
        if to_call > 0:
            action = "call"
            amount = min(to_call, my_chips)
    elif action == "call":
        amount = min(to_call, my_chips)
    elif action == "raise":
        if raise_cap_reached:
            action = "call" if to_call > 0 else "check"
            amount = min(to_call, my_chips) if action == "call" else 0
        else:
            desired_total_bet = my_bet + amount
            min_legal_total_bet = min_raise_to
            if my_chips + my_bet < min_legal_total_bet:
                action = "call" if to_call > 0 else "check"
                amount = min(to_call, my_chips) if action == "call" else 0
            else:
                desired_total_bet = max(desired_total_bet, min_legal_total_bet)
                desired_total_bet = min(desired_total_bet, my_bet + my_chips)
                amount = max(0, desired_total_bet - my_bet)
    else:
        action = "check" if to_call == 0 else "call"
        amount = 0 if action == "check" else min(to_call, my_chips)

    chips_after_action = my_chips - amount
    print_player_action(name, action, amount, speech , chips_after_action, hole_cards)

    new_players_info = dict(state["players_info"])
    new_players_info[name] = dict(new_players_info[name])
    new_players_info[name]["action"] = action
    new_players_info[name]["bet"] = new_players_info[name].get("bet", 0) + amount
    new_players_info[name]["total_bet"] = new_players_info[name].get("total_bet", 0) + amount
    if action == "fold":
        new_players_info[name]["folded"] = True

    new_raises_in_street = raises_in_street
    new_last_raise_size = last_raise_size
    if action == "raise":
        new_total_bet = new_players_info[name].get("bet", 0)
        new_last_raise_size = max(min_open_raise, new_total_bet - current_max_bet)
        new_raises_in_street = raises_in_street + 1
        # raise 时，重置其他仍可行动玩家的 action，让他们重新行动
        for other in order:
            if other == name:
                continue
            if _is_active(other, new_players_info, state["chips"]):
                other_info = dict(new_players_info[other])
                other_info["action"] = ""
                new_players_info[other] = other_info

    new_pot = state["pot"] + amount
    new_chips = dict(state["chips"])
    new_chips[name] -= amount

    active = [n for n in order if _is_active(n, new_players_info, new_chips)]
    street_finished = _street_bets_balanced(new_players_info, active)

    return {
        "pot": new_pot,
        "chips": new_chips,
        "players_info": new_players_info,
        "current_player": next_name,
        "street_finished": street_finished,
        "raises_in_street": new_raises_in_street,
        "last_raise_size": new_last_raise_size,
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

    if state.get("street_finished", False):
        if state["stage"] == "river":
            return "judge"
        return "deal_community"

    # 所有未 fold 的人都 all-in（active 为空），本街无人可行动，直接进下一街
    if len(active) == 0:
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
