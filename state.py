"""
state.py — 游戏状态定义

【核心概念：LangGraph 的 State】

LangGraph 的本质是一个「状态机」：
  - 图里的每个「节点（Node）」接收当前 State，返回对 State 的「局部更新」
  - 框架负责把更新合并回全局 State，再决定走哪条「边（Edge）」
  - 你永远不需要手动传递变量，State 就是整个系统的共享黑板

这里用 TypedDict 定义 State，好处：
  1. 静态类型检查（mypy/pyright 能发现字段名拼错）
  2. IDE 自动补全
  3. LangGraph 内部用它做 schema 校验

【两层结构的原因】
  PlayerState（TypedDict）= 快照/序列化友好，可以 JSON 存取，存进 GameState
  Player（dataclass）     = 有行为方法，游戏逻辑层用它操作
  节点函数：Player.from_state() 把快照还原成对象 → 操作 → to_state() 存回去
"""
from __future__ import annotations
# Annotated 让我们给类型附加「元数据」，LangGraph 用它识别如何合并字段
from typing import Annotated, Any
from typing_extensions import TypedDict
import operator  # operator.add 就是列表的 + 合并，用于行动历史追加


class PlayerState(TypedDict):
    """
    单个玩家的「状态快照」，以字典形式嵌入 GameState.players 列表中。

    【为什么不直接用 Player 对象？】
    LangGraph 的 State 需要是「可序列化」的（支持 JSON / pickle），
    用于 checkpointer 持久化（保存进度、断点续传、历史回放）。
    dataclass 对象不能直接 JSON 序列化，但 TypedDict（纯字典）可以。
    所以：游戏逻辑层用 Player 对象，存储层用 PlayerState 字典。
    """
    id: int            # 玩家唯一编号（0-based，不随局变化）
    name: str          # 显示名 
    chips: int         # 当前筹码量（归零即淘汰）
    hole_cards: list[str]   # 底牌，如 ['Ah', 'Kd']；字符串格式：Rank+Suit
    is_active: bool         # False = 已 fold，本局不再参与
    is_all_in: bool         # True = 筹码已全押，不再行动但仍参与摊牌
    is_human: bool          # True = 人类玩家，遇到此玩家需 interrupt 等待输入
    is_ai: bool             # True = 由 LLM 控制
    current_street_bet: int # 「本街道」已投入金额（flop/turn/river 开始时重置为 0）
    total_bet: int          # 「本局」累计投入金额（用于边池计算）
    position: int           # 座位索引（0-based，决定翻前行动顺序）
    player_type: str        # AI 人设：'aggressive'|'passive'|'balanced'|'bluffer'|'math'
    # ── 跨局统计（可选，用于对手建模）────────────────────────────
    vpip: float   # VPIP = Voluntarily Put money In Pot，自愿入池率，衡量松紧程度
    pfr: float    # PFR  = Pre-Flop Raise rate，翻前加注率，衡量主动程度
    # ── 上帝视角字段 ────────────────────────────────────────────
    thought: str  # 本次决策的内心独白（LLM 输出的 thought 字段），观战时可见


class GameState(TypedDict):
    """
    整局游戏的全局状态，是 LangGraph 图里流转的「共享黑板」。

    【LangGraph State 更新机制】
    - 节点函数返回一个「部分字典」，只包含本节点修改的字段
    - LangGraph 默认用「覆盖」合并：新值直接替换旧值
    - 特殊情况：用 Annotated[T, reducer] 指定自定义合并函数
      例如 action_history 用 operator.add，表示「追加」而不是「覆盖」
      这样每个节点只需返回「新增的行动」，框架自动拼接完整历史
    """

    # ── 牌局基础信息 ──────────────────────────────────────────────────────
    players: list[PlayerState]   # 所有玩家快照列表，索引即座位号
    deck: list[str]              # 剩余牌堆（已发出的牌从列表中 pop 移除）
    community_cards: list[str]   # 公共牌，preflop=[], flop=3张, turn=4张, river=5张
    pot: int                     # 主底池筹码总量
    side_pots: list[dict]        # 边池（有玩家 all-in 时产生），格式：
                                 # [{'amount': int, 'eligible': [player_id, ...]}]
    current_bet: int             # 本街道当前最高下注额（跟注须达到此值）
    min_raise: int               # 最小加注额（上一次加注幅度，防止无限小额加注）

    # ── 位置与行动顺序 ────────────────────────────────────────────────────
    dealer_position: int         # 庄家（Button）座位索引，每局顺时针轮转
    small_blind: int             # 小盲注金额（通常 = big_blind / 2）
    big_blind: int               # 大盲注金额（基准下注单位）
    current_player_index: int    # 「当前行动玩家」在 players 列表中的索引
    action_order: list[int]      # 本街道行动顺序（player id 序列）

    # ── 街道（Street）────────────────────────────────────────────────────
    # 德州扑克四个街道：翻前(preflop) → 翻牌(flop) → 转牌(turn) → 河牌(river)
    street: str                  # 当前街道名
    street_action_count: int     # 本街道累计行动次数（判断一圈是否完成）

    # ── 行动历史（使用 reducer 追加，而非覆盖）────────────────────────────
    # Annotated[list[dict], operator.add] 告诉 LangGraph：
    # 当多个节点都更新此字段时，用 list + list 追加，而非后者覆盖前者
    # 这是 LangGraph「自定义 reducer」的核心用法
    action_history: Annotated[list[dict], operator.add]

    # ── AI 决策中间结果 ───────────────────────────────────────────────────
    agent_thoughts: dict[str, str]  # 上帝视角：{玩家名: 本轮思考内容}
    agent_decision: dict | None     # 当前 AI 玩家的决策，execute_action 后清空

    # ── 人类玩家输入 ──────────────────────────────────────────────────────
    # interrupt() 暂停图执行后，外部调用 graph.update_state() 写入此字段
    # 格式：{'action': 'fold'|'call'|'raise'|'check', 'amount': int}
    human_action: dict | None
    # 人类玩家对话输入（chat interrupt 后由外部写入）
    human_chat: str | None

    # ── 对话系统 ─────────────────────────────────────────────────────────
    # 每条消息格式：
    #   {'player_name': str, 'message': str, 'is_bluff': bool, 'street': str}
    # is_bluff 只在上帝视角可见，标识这句话是否带有欺骗意图
    # 同样使用 operator.add reducer，每轮对话追加而非覆盖
    chat_history: Annotated[list[dict], operator.add]
    # pending_chat：当前街道对话阶段的临时缓冲，收集完所有人发言后清空
    pending_chat: list[dict]
    # chat_round_index：记录本街道已经发言的玩家数，驱动对话轮次循环
    chat_round_index: int

    # ── 控制流字段 ────────────────────────────────────────────────────────
    round_number: int       # 当前局数（每局 +1，用于显示和统计）
    game_over: bool         # True = 游戏彻底结束（只剩1名有筹码玩家）
    winners: list[int]      # 本局赢家 player id 列表（平局时多人）
    winner_message: str     # 人类可读的结算描述，如「闪电赢得底池 200 筹码」
    error_message: str      # 节点运行错误信息（调试用），正常情况为空字符串
