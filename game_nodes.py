# -*- coding: utf-8 -*-
"""
LangGraph 游戏节点定义
玩家驱动方式：单一 player_action 节点，通过 current_player 轮转
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
    current_player: str          # 当前行动的玩家名
    player_order: list           # 玩家顺序列表，决定轮转
    stage: str
    hand_ended: bool
    winner: str
    round_count: int


# ══════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════

def _is_active(name: str, players_info: dict, chips: dict) -> bool:
    """判断玩家是否仍在局中（未弃牌且有筹码）"""
    if players_info.get(name, {}).get("folded", False):
        return False
    if chips.get(name, 0) <= 0:
        return False
    return True


def _next_player(current: str, order: list) -> str:
    """返回下一个玩家名，循环轮转"""
    idx = order.index(current)
    return order[(idx + 1) % len(order)]


def _is_last_player(current: str, order: list) -> bool:
    """判断当前玩家是否是本轮最后一个"""
    return order.index(current) == len(order) - 1


def _check_hand_ended(state: GameState) -> bool:
    """统一判断本手牌是否应结束"""
    players_info = state["players_info"]
    chips = state["chips"]
    order = state["player_order"]
    active = [n for n in order if _is_active(n, players_info, chips)]
    if len(active) <= 1:
        return True
    all_checked = all(players_info[n].get("action") == "check" for n in active)
    if all_checked:
        return True
    if state["round_count"] >= 3:
        return True
    return False


# ══════════════════════════════════════════════════════════════
# 发牌节点
# ══════════════════════════════════════════════════════════════

def dealer_node(state: GameState) -> dict:
    """发牌节点"""
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
        "community_cards": [],
        "pot": 0,
        "stage": "preflop",
        "current_player": order[0],
        "hand_ended": False,
        "winner": "",
        "round_count": 1,
    }


# ══════════════════════════════════════════════════════════════
# 通用玩家行动节点（所有玩家共用）
# ══════════════════════════════════════════════════════════════

def player_action_node(state: GameState) -> dict:
    """通用玩家行动节点，根据 current_player 决定谁行动"""
    name = state["current_player"]
    order = state["player_order"]
    next_name = _next_player(name, order)
    is_last = _is_last_player(name, order)

    # 玩家不在局中（弃牌或出局），直接跳过
    if not _is_active(name, state["players_info"], state["chips"]):
        new_round = state["round_count"] + 1 if is_last else state["round_count"]
        new_state = {**state, "round_count": new_round}
        hand_ended = _check_hand_ended(new_state) if is_last else False
        return {
            "current_player": next_name,
            "round_count": new_round,
            "hand_ended": hand_ended,
            "messages": [],
        }

    hole_cards = state["players_info"][name]["hole_cards"]
    my_chips = state["chips"].get(name, 0)

    # 计算下注信息
    current_max_bet = max(p.get("bet", 0) for p in state["players_info"].values())
    my_bet = state["players_info"][name].get("bet", 0)
    to_call = current_max_bet - my_bet

    # 构建 game_context
    game_context = "对手行动：\n"
    for other_name in order:
        if other_name == name:
            continue
        other_info = state["players_info"].get(other_name, {})
        if other_info.get("action"):
            game_context += f"  {other_name}：{other_info['action']}（下注{other_info.get('bet', 0)}）\n"
    game_context += f"\n当前底池：{state['pot']}\n"
    game_context += f"你当前筹码：{my_chips}（这是你能下注的上限，不能超过此数）\n"
    game_context += f"你本轮已下注：{my_bet}，当前最高下注：{current_max_bet}，需跟注金额：{to_call}\n"
    if to_call == 0:
        game_context += "提示：当前无人下注，to_call=0，check 零成本合法，fold 为劣后选择，请优先考虑 check 或 raise。\n"
    else:
        game_context += f"提示：跟注需 {to_call} 筹码，可选 call/raise/fold。\n"

    # 获取行动：AI 或真人
    is_human = state["players_info"][name].get("is_human", False)
    if is_human:
        action, amount, speech = _human_input(name, hole_cards, game_context, to_call, my_chips)
    else:
        action, amount, speech = call_llm_for_action(name, hole_cards, game_context)

    # 强制限制下注额
    amount = max(0, min(amount, my_chips))

    print_player_action(name, action, amount, speech)

    # 更新 players_info（局部修改后整体写回）
    new_players_info = dict(state["players_info"])
    new_players_info[name] = dict(new_players_info[name])
    new_players_info[name]["action"] = action
    new_players_info[name]["bet"] = amount
    new_players_info[name]["total_bet"] = new_players_info[name].get("total_bet", 0) + amount
    if action == "fold":
        new_players_info[name]["folded"] = True

    new_pot = state["pot"] + amount
    new_chips = dict(state["chips"])
    new_chips[name] -= amount

    # 如果是本轮最后一个玩家，更新轮次并判断结束
    if is_last:
        new_round = state["round_count"] + 1
        hand_ended = _check_hand_ended({
            **state,
            "players_info": new_players_info,
            "chips": new_chips,
            "round_count": new_round,
        })
    else:
        new_round = state["round_count"]
        hand_ended = False

    return {
        "pot": new_pot,
        "chips": new_chips,
        "players_info": new_players_info,
        "current_player": next_name,
        "round_count": new_round,
        "hand_ended": hand_ended,
        "messages": [{"agent": name, "action": action, "speech": speech}],
    }


def _human_input(name: str, hole_cards: list, game_context: str, to_call: int, my_chips: int) -> tuple:
    """真人输入行动（预留接口）"""
    cards_str = " ".join([str(c) for c in hole_cards])
    print(f"\n你的手牌：{cards_str}")
    print(game_context)
    print("可选行动：fold / check / call / raise <金额>")
    raw = input(f"{name} 请输入行动：").strip()
    parts = raw.split()
    action = parts[0].lower() if parts else "check"
    amount = int(parts[1]) if len(parts) > 1 else 0
    if action not in ["fold", "check", "call", "raise"]:
        action = "check"
    if action in ["fold", "check"]:
        amount = 0
    speech = ""
    return (action, amount, speech)


# ══════════════════════════════════════════════════════════════
# 判断赢家节点
# ══════════════════════════════════════════════════════════════

def judge_winner_node(state: GameState) -> dict:
    """判断赢家并结算筹码"""
    winner = determine_winner(state["players_info"], state["community_cards"])

    # 筹码已在每次行动时实时扣除，这里只需给赢家加回底池
    new_chips = dict(state["chips"])

    if winner is None:
        print_winner_info("无人获胜", state["pot"], new_chips)
        return {
            "winner": "",
            "chips": new_chips,
            "hand_ended": True,
        }

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
