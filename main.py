# -*- coding: utf-8 -*-
"""德州扑克 AI Agent 入口。

用法:
    python main.py              # 纯 AI 对战（观战模式）
    python main.py --human      # 人机对战
    python main.py --human --max-rounds 20
"""
from __future__ import annotations
import argparse
import os
import sys
import uuid
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from graph import build_graph, make_initial_state

console = Console()

DEFAULT_AI_PLAYERS = [
    {"name": "闪电(激进)", "chips": 1000, "player_type": "aggressive", "is_ai": True},
    {"name": "磐石(保守)", "chips": 1000, "player_type": "passive",    "is_ai": True},
    {"name": "均衡者(GTO)", "chips": 1000, "player_type": "balanced",   "is_ai": True},
    {"name": "幻影(诈唬)", "chips": 1000, "player_type": "bluffer",    "is_ai": True},
]


def print_state(state: dict, show_all_cards: bool = False) -> None:
    console.rule(f"[bold cyan]? {state.get('round_number','?')} ? -- {state.get('street','').upper()}")
    community = state.get("community_cards", [])
    c_str = "  ".join(f"[bold yellow]{c}[/]" for c in community) if community else "[dim](翻牌前)[/]"
    console.print(f"公共牌: {c_str}    底池: [bold green]{state.get('pot', 0)}[/]")
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("玩家", min_width=14)
    table.add_column("筹码", justify="right")
    table.add_column("本街下注", justify="right")
    table.add_column("手牌", min_width=8)
    table.add_column("状态")
    table.add_column("内心独白", min_width=28)

    thoughts = state.get("agent_thoughts", {})
    players = state.get("players", [])
    cur = state.get("current_player_index", -1)

    for i, p in enumerate(players):
        name = p["name"]
        name_str = f"[bold cyan]> {name}[/]" if i == cur else name
        if p["is_human"] or show_all_cards:
            cards_str = " ".join(f"[bold yellow]{c}[/]" for c in p.get("hole_cards", []))
        else:
            cards_str = "[dim]🂠 🂠[/]"
        if not p["is_active"]:
            status = "[red]弃牌[/]"
        elif p["is_all_in"]:
            status = "[magenta]全押[/]"
        else:
            status = "[green]活跃[/]"
        thought = thoughts.get(name, "")
        t_short = thought[:38] + "..." if len(thought) > 38 else thought
        table.add_row(name_str, str(p["chips"]), str(p["current_street_bet"]),
                      cards_str, status, f"[dim]{t_short}[/]")
    console.print(table)


def print_winner(state: dict) -> None:
    msg = state.get("winner_message", "")
    if msg:
        console.print(Panel(f"[bold gold1]{msg}[/]", title="[bold]结算", border_style="gold1"))


def print_chat_message(entry: dict, god_view: bool = True) -> None:
    name = entry.get("player_name", "?")
    message = entry.get("message", "")
    is_bluff = entry.get("is_bluff", False)
    inner = entry.get("inner_reason", "")
    if not message:
        console.print(f"  [dim]{name} 沉默[/]")
        return
    bluff_tag = " [bold red][诈唬][/]" if (god_view and is_bluff) else ""
    console.print(f"  [bold]{name}[/]: [italic]\"{message}\"{bluff_tag}")
    if god_view and inner and inner != "(人类玩家)":
        console.print(f"    [dim italic]内心: {inner}[/]")


def get_human_chat(state: dict) -> str:
    players = state["players"]
    human = next((p for p in players if p.get("is_human")), None)
    if not human:
        return ""
    community = state.get("community_cards", [])
    street = state.get("street", "")
    console.print(f"\n[bold cyan]── 对话阶段 [{street.upper()}] --[/]")
    if community:
        console.print(f"公共牌: {' '.join(community)}")
    console.print(f"你的手牌: [bold yellow]{' '.join(human.get('hole_cards', []))}[/]")
    for msg in state.get("chat_history", []):
        if msg.get("street") == street and msg.get("message"):
            console.print(f"  [cyan]{msg['player_name']}[/]: \"{msg['message']}\"")
    try:
        raw = console.input("[bold]你说（直接回车=沉默）: [/]").strip()
    except (EOFError, KeyboardInterrupt):
        raw = ""
    return raw


def get_human_action(state: dict) -> dict:
    players = state["players"]
    idx = state["current_player_index"]
    player = players[idx]
    current_bet = state["current_bet"]
    my_bet = player["current_street_bet"]
    call_amount = max(0, current_bet - my_bet)
    my_chips = player["chips"]
    min_raise = state["min_raise"]

    console.print(f"\n[bold cyan]用法:[/] {' '.join(player['hole_cards'])}"
                  f"  筹码: [green]{my_chips}[/]  跟注需要: [yellow]{call_amount}[/]")

    valid_actions = ["fold", "call", "raise"]
    if call_amount == 0:
        valid_actions = ["check", "fold", "raise"]

    while True:
        action_str = "/".join(valid_actions)
        try:
            raw = console.input(f"你的行动 [{action_str}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return {"action": "fold", "amount": 0}
        if raw not in valid_actions:
            console.print(f"[red]无效输入，请输入 {action_str}[/]")
            continue
        amount = 0
        if raw == "raise":
            max_raise = my_bet + my_chips
            while True:
                try:
                    amt_str = console.input(
                        f"加注到多少（（最小 {current_bet + min_raise}，最大 {max_raise}）: "
                    ).strip()
                    amount = int(amt_str)
                    if current_bet + min_raise <= amount <= max_raise:
                        break
                    console.print("[red]金额不合法[/]")
                except ValueError:
                    console.print("[red]请输入整数[/]")
        return {"action": raw, "amount": amount}


def run(human_mode: bool = False, max_rounds: int = 50) -> None:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        # 尝试加载 .env 文件
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

    if not os.environ.get("DASHSCOPE_API_KEY"):
        console.print("[bold red]错误：未设置 DASHSCOPE_API_KEY 环境变量[/]")
        console.print("请执行: set DASHSCOPE_API_KEY=your_key_here  (Windows)")
        sys.exit(1)

    player_configs = list(DEFAULT_AI_PLAYERS)
    if human_mode:
        player_configs.append({
            "name": "你(人类)", "chips": 1000,
            "player_type": "balanced", "is_human": True, "is_ai": False,
        })

    graph = build_graph()
    initial_state = make_initial_state(player_configs)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    console.print(Panel(
        "[bold green]德州扑克 AI Agent 启动[/]\n" +
        "  ".join(f"[cyan]{p['name']}[/]" for p in player_configs),
        title="♠ Texas Hold'em", border_style="green",
    ))

    current_state = initial_state
    rounds_played = 0
    god_view = True
    # 去重集合：记录已打印过的 (player_name, message, street) 和行动历史长度
    printed_chat_ids: set = set()
    printed_action_count = 0

    while rounds_played < max_rounds:
        try:
            for event in graph.stream(current_state, config=config, stream_mode="values"):
                current_state = event

                interrupts = current_state.get("__interrupt__")
                if interrupts:
                    payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
                    if isinstance(payload, dict) and "waiting_for_chat" in payload:
                        chat_text = get_human_chat(current_state)
                        graph.update_state(config, {"human_chat": chat_text})
                    else:
                        print_state(current_state, show_all_cards=god_view)
                        human_act = get_human_action(current_state)
                        graph.update_state(config, {"human_action": human_act})
                    continue

                # ── 打印新增的对话消息（去重）──────────────────────────────
                for entry in current_state.get("chat_history", []):
                    chat_id = (
                        entry.get("player_name", ""),
                        entry.get("message", ""),
                        entry.get("street", ""),
                        entry.get("is_silence", False),
                    )
                    if chat_id not in printed_chat_ids:
                        printed_chat_ids.add(chat_id)
                        print_chat_message(entry, god_view=god_view)

                # ── 打印新增的行动（去重）──────────────────────────────────
                history = current_state.get("action_history", [])
                for act_entry in history[printed_action_count:]:
                    printed_action_count += 1
                    act = act_entry.get("action", "")
                    pname = act_entry.get("player_name", "")
                    amt = act_entry.get("amount", 0)
                    street = act_entry.get("street", "")
                    if act not in ("small_blind", "big_blind", "showdown"):
                        console.print(
                            f"  [dim]{street}[/] [bold]{pname}[/]: "
                            f"[yellow]{act}[/]" + (f" {amt}" if amt else "")
                        )

                # 公共牌更新时打印局面（board_updated 信号）
                if current_state.get("board_updated"):
                    community = current_state.get("community_cards", [])
                    street = current_state.get("street", "").upper()
                    c_str = "  ".join(f"[bold yellow]{c}[/]" for c in community)
                    console.rule(f"[bold cyan]── {street} ──")
                    console.print(f"公共牌: {c_str}    底池: [bold green]{current_state.get('pot',0)}[/]")
                    if god_view:
                        for p in current_state.get("players", []):
                            if p.get("hole_cards"):
                                cards = " ".join(f"[yellow]{c}[/]" for c in p["hole_cards"])
                                active_str = "" if p["is_active"] else " [dim](已弃牌)[/]"
                                console.print(f"  [dim]{p['name']}:[/] {cards}{active_str}")
                    console.print()

                if current_state.get("winner_message"):
                    run._last_community = []  # 重置，新局重新打印
                    printed_chat_ids.clear()
                    printed_action_count = 0
                    print_state(current_state, show_all_cards=True)
                    print_winner(current_state)
                    rounds_played += 1

                if current_state.get("game_over"):
                    console.rule("[bold red]游戏结束")
                    alive = [p for p in current_state["players"] if p["chips"] > 0]
                    if alive:
                        console.print(f"[bold gold1]最终赢家: {alive[0]['name']}  筹码: {alive[0]['chips']}[/]")
                    return

        except KeyboardInterrupt:
            console.print("\n[yellow]游戏已中止[/]")
            return
        except Exception as e:
            console.print(f"[bold red]运行错误: {e}[/]")
            import traceback
            traceback.print_exc()
            return

    console.print(f"[yellow]已达到最大局数 {max_rounds}，游戏结束[/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="德州扑克 AI Agent")
    parser.add_argument("--human", action="store_true", help="加入人类玩家（人机对战）")
    parser.add_argument("--max-rounds", type=int, default=50)
    args = parser.parse_args()
    run(human_mode=args.human, max_rounds=args.max_rounds)
