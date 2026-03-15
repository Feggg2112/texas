"""游戏逻辑节点：初始化、发牌、执行行动、街道推进、结算等。"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.player import Player
from utils.poker_utils import (
    make_deck,
    deal_hole_cards,
    deal_flop,
    deal_one,
    determine_winners,
    cards_to_str,
)

STREET_ORDER = ["preflop", "flop", "turn", "river"]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _active_can_act(players: list[dict]) -> list[dict]:
    """返回仍可行动的玩家（活跃且未 all-in）。"""
    return [p for p in players if p["is_active"] and not p["is_all_in"]]


def _find_first_active(players: list[dict], start: int) -> int:
    """从 start 位置起找下一个活跃且可行动的玩家索引。"""
    n = len(players)
    for i in range(n):
        idx = (start + i) % n
        if players[idx]["is_active"] and not players[idx]["is_all_in"]:
            return idx
    return start


# ── 节点1：initialize_game ────────────────────────────────────────────────────

def initialize_game(state: dict) -> dict:
    """
    初始化 / 开始新一局：洗牌、重置玩家、发手牌、收盲注。
    期望 state 中已有 players 列表（含 id/name/chips/player_type/is_human）。
    """
    players_raw = state["players"]
    round_number = state.get("round_number", 0) + 1
    small_blind = state.get("small_blind", 10)
    big_blind = state.get("big_blind", 20)
    n = len(players_raw)

    # 重置玩家状态
    players = [Player.from_state(p) for p in players_raw]
    for p in players:
        p.reset_for_new_round()

    # 决定庄家位置（轮转）
    old_dealer = state.get("dealer_position", -1)
    dealer_pos = (old_dealer + 1) % n
    for _ in range(n):
        if players[dealer_pos].chips > 0:
            break
        dealer_pos = (dealer_pos + 1) % n

    sb_pos = (dealer_pos + 1) % n
    bb_pos = (dealer_pos + 2) % n

    # 洗牌发牌
    deck = make_deck()
    hole_cards_list, deck = deal_hole_cards(deck, n)
    for i, p in enumerate(players):
        p.hole_cards = hole_cards_list[i]

    # 收盲注
    sb_actual = players[sb_pos].post_blind(small_blind)
    bb_actual = players[bb_pos].post_blind(big_blind)
    pot = sb_actual + bb_actual

    # 翻前行动顺序：从 UTG（bb 后一位）开始
    utg_pos = (bb_pos + 1) % n
    action_order = []
    idx = utg_pos
    for _ in range(n):
        if players[idx].chips > 0:
            action_order.append(players[idx].id)
        idx = (idx + 1) % n

    action_history = [
        {"player_id": players[sb_pos].id, "player_name": players[sb_pos].name,
         "action": "small_blind", "amount": sb_actual, "street": "preflop"},
        {"player_id": players[bb_pos].id, "player_name": players[bb_pos].name,
         "action": "big_blind", "amount": bb_actual, "street": "preflop"},
    ]

    return {
        "players": [p.to_state() for p in players],
        "deck": deck,
        "community_cards": [],
        "pot": pot,
        "side_pots": [],
        "current_bet": bb_actual,
        "min_raise": big_blind,
        "dealer_position": dealer_pos,
        "small_blind": small_blind,
        "big_blind": big_blind,
        "current_player_index": utg_pos,
        "action_order": action_order,
        "street": "preflop",
        "street_action_count": 0,
        "action_history": action_history,
        "agent_thoughts": {},
        "human_action": None,
        "round_number": round_number,
        "game_over": False,
        "winners": [],
        "winner_message": "",
        "error_message": "",
    }


# ── 节点2：deal_community_cards ───────────────────────────────────────────────

def deal_community_cards(state: dict) -> dict:
    """根据当前街道发公共牌（flop/turn/river）。"""
    street = state["street"]
    deck = list(state["deck"])
    community = list(state["community_cards"])

    if street == "flop":
        new_cards, deck = deal_flop(deck)
        community.extend(new_cards)
    elif street in ("turn", "river"):
        card, deck = deal_one(deck)
        community.append(card)
    # preflop 不发公共牌，直接返回

    return {"community_cards": community, "deck": deck}


# ── 节点3：execute_action ────────────────────────────────────────────────────

def execute_action(state: dict) -> dict:
    """
    执行当前玩家的行动。
    - AI 玩家：行动来自 state['agent_decision']（由 ai_nodes 写入）
    - 人类玩家：行动来自 state['human_action']（由外部 interrupt 写入）
    """
    players = [Player.from_state(p) for p in state["players"]]
    idx = state["current_player_index"]
    current_player = players[idx]
    street = state["street"]
    current_bet = state["current_bet"]
    pot = state["pot"]
    min_raise = state["min_raise"]
    big_blind = state["big_blind"]

    # 获取行动指令
    if current_player.is_human:
        raw = state.get("human_action") or {"action": "fold", "amount": 0}
    else:
        raw = state.get("agent_decision") or {"action": "fold", "amount": 0}

    action = raw.get("action", "fold")
    amount = int(raw.get("amount", 0))

    # 执行行动
    call_amount = max(0, current_bet - current_player.current_street_bet)

    if action == "fold":
        action_record = current_player.fold()
    elif action == "check":
        if call_amount > 0:
            # 无法过牌，强制跟注
            action_record = current_player.call(current_bet)
            action = "call"
        else:
            action_record = current_player.check()
    elif action == "call":
        action_record = current_player.call(current_bet)
    elif action == "raise":
        # amount 是加注到的总额（本街道）
        raise_to = max(amount, current_bet + min_raise)
        raise_to = min(raise_to, current_player.current_street_bet + current_player.chips)
        action_record = current_player.raise_bet(raise_to)
        # 更新最高下注和最小再加注
        new_raise_size = raise_to - current_bet
        current_bet = raise_to
        min_raise = max(big_blind, new_raise_size)
    else:
        action_record = current_player.fold()

    # 更新底池
    pot += action_record["amount"]

    # 记录行动
    action_record["player_name"] = current_player.name
    action_record["street"] = street

    # 找下一个可行动玩家
    n = len(players)
    next_idx = (idx + 1) % n
    for _ in range(n):
        p = players[next_idx]
        if p.is_active and not p.is_all_in:
            break
        next_idx = (next_idx + 1) % n

    return {
        "players": [p.to_state() for p in players],
        "pot": pot,
        "current_bet": current_bet,
        "min_raise": min_raise,
        "current_player_index": next_idx,
        "street_action_count": state["street_action_count"] + 1,
        "action_history": [action_record],
        "human_action": None,
        "agent_decision": None,
    }


# ── 节点4：advance_street ────────────────────────────────────────────────────

def advance_street(state: dict) -> dict:
    """推进到下一街道：重置玩家本街下注，确定新行动顺序。"""
    current_street = state["street"]
    idx = STREET_ORDER.index(current_street)
    next_street = STREET_ORDER[idx + 1] if idx + 1 < len(STREET_ORDER) else "showdown"

    players = [Player.from_state(p) for p in state["players"]]
    for p in players:
        p.reset_for_new_street()

    # 新街道行动顺序：从庄家左手边第一个活跃玩家开始
    dealer = state["dealer_position"]
    n = len(players)
    first_idx = _find_first_active([p.to_state() for p in players], (dealer + 1) % n)

    return {
        "players": [p.to_state() for p in players],
        "street": next_street,
        "current_bet": 0,
        "min_raise": state["big_blind"],
        "current_player_index": first_idx,
        "street_action_count": 0,
    }


# ── 节点5：showdown ───────────────────────────────────────────────────────────

def showdown(state: dict) -> dict:
    """摊牌：比较所有活跃玩家手牌，决出赢家。"""
    players = state["players"]
    community = state["community_cards"]

    active = [
        {"id": p["id"], "hole_cards": p["hole_cards"]}
        for p in players if p["is_active"]
    ]

    winner_ids = determine_winners(active, community)

    # 分配底池
    pot = state["pot"]
    share = pot // len(winner_ids) if winner_ids else 0
    remainder = pot % len(winner_ids) if winner_ids else 0

    updated_players = []
    winner_names = []
    for p in players:
        player = Player.from_state(p)
        if player.id in winner_ids:
            player.chips += share
            winner_names.append(player.name)
        updated_players.append(player.to_state())

    # 余数给第一个赢家
    if remainder and winner_ids:
        for p in updated_players:
            if p["id"] == winner_ids[0]:
                p["chips"] += remainder
                break

    winner_msg = f"赢家: {', '.join(winner_names)}，赢得底池 {pot} 筹码"

    return {
        "players": updated_players,
        "winners": winner_ids,
        "winner_message": winner_msg,
        "pot": 0,
        "action_history": [{"action": "showdown", "street": "showdown",
                            "winners": winner_names, "pot": pot}],
    }


# ── 节点6：check_game_over ────────────────────────────────────────────────────

def check_game_over(state: dict) -> dict:
    """检查是否还有多于 1 名有筹码的玩家，决定游戏是否结束。"""
    players = state["players"]
    alive = [p for p in players if p["chips"] > 0]
    game_over = len(alive) <= 1
    return {"game_over": game_over}


# ── 路由函数（供 graph.py 使用） ──────────────────────────────────────────────

def route_after_action(state: dict) -> str:
    """
    execute_action 后的路由：
    - 只剩 1 名活跃玩家 → 'only_one_active'（直接结算）
    - 本街道下注轮结束  → 'street_complete'
    - 否则继续          → 'continue_betting'
    """
    players = state["players"]
    active = [p for p in players if p["is_active"]]

    if len(active) <= 1:
        return "only_one_active"

    # 判断本街道是否结束：所有可行动玩家的下注额相等，且至少每人都行动过一次
    can_act = _active_can_act(players)
    if not can_act:
        return "street_complete"

    max_bet = max(p["current_street_bet"] for p in active)
    all_matched = all(p["current_street_bet"] == max_bet for p in can_act)

    # 至少需要 len(can_act) 次行动才算一圈
    min_actions_needed = len([p for p in players if p["is_active"]])
    if all_matched and state["street_action_count"] >= min_actions_needed:
        return "street_complete"

    return "continue_betting"


def route_after_street(state: dict) -> str:
    """advance_street 后的路由：下一街道是否为 showdown。"""
    return "showdown" if state["street"] == "showdown" else "deal_cards"


def route_game_over(state: dict) -> str:
    """check_game_over 后的路由。"""
    return "end" if state["game_over"] else "new_round"


def route_current_player(state: dict) -> str:
    """判断当前玩家是人类还是 AI，决定走哪个决策节点。"""
    players = state["players"]
    idx = state["current_player_index"]
    if players[idx]["is_human"]:
        return "human_input"
    return "ai_decision"
