"""
models/player.py - Player 运行时对象

【设计思路：两层模型分离】

  状态层 PlayerState (TypedDict)  -> 纯字典，可序列化，存进 LangGraph State
  对象层 Player (dataclass)       -> 带方法，游戏逻辑用它执行 fold/call/raise

  转换关系：
    Player.from_state(dict)  -> 从 State 快照还原成对象
    player.to_state()        -> 把对象序列化回字典存回 State

  为什么这样设计？
  LangGraph 的 checkpointer 需要把 State 序列化（存磁盘/Redis），
  dataclass 对象无法直接 JSON 化，但纯字典可以。
  所以「存储用字典，操作用对象」，进节点时反序列化，出节点时序列化。
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Player:
    """
    运行时 Player 对象。

    【@dataclass 的作用】
    自动生成 __init__、__repr__、__eq__ 等方法，
    让我们只需声明字段，不需要手写构造函数。
    field(default_factory=list) 确保每个玩家有独立的列表实例，
    避免所有玩家共享同一个列表的经典 Python 陷阱。
    """
    id: int
    name: str
    chips: int
    hole_cards: list[str] = field(default_factory=list)
    is_active: bool = True        # 本局是否还活着（未 fold）
    is_all_in: bool = False
    is_human: bool = False
    is_ai: bool = True
    current_street_bet: int = 0   # 本街道已下注额（每条街道开始时重置）
    total_bet: int = 0            # 本局累计下注额（用于边池计算）
    position: int = 0             # 座位索引（0-based）
    player_type: str = "balanced" # AI 人设类型
    vpip: float = 0.0             # 自愿入池率（Voluntarily Put In Pot）
    pfr: float = 0.0              # 翻前加注率（Pre-Flop Raise）
    thought: str = ""             # 本次决策内心独白（上帝视角可见）

    # ── @property：计算属性，不存储值，每次访问时实时计算 ─────────────────

    @property
    def is_broke(self) -> bool:
        """筹码耗尽，下局无法参赛。"""
        return self.chips <= 0

    @property
    def can_act(self) -> bool:
        """
        是否还能在本街道行动。
        is_active=False（已 fold）或 is_all_in=True（已全押）的玩家
        本街道跳过，不参与行动循环。
        """
        return self.is_active and not self.is_all_in

    # ── 行动方法：每个方法执行一种扑克行动，更新自身状态并返回行动记录 ───
    # 返回的字典会被追加到 GameState.action_history

    def fold(self) -> dict:
        """弃牌：退出本局，is_active 置 False。"""
        self.is_active = False
        return {"player_id": self.id, "action": "fold", "amount": 0}

    def call(self, current_bet: int) -> dict:
        """
        跟注：补齐到 current_bet。
        min() 处理筹码不足的情况（自动变成 all-in）。
        """
        # 只需补齐「差额」，而不是付出 current_bet 全额
        amount = min(current_bet - self.current_street_bet, self.chips)
        self.chips -= amount
        self.current_street_bet += amount
        self.total_bet += amount
        if self.chips == 0:
            self.is_all_in = True  # 筹码恰好用完 = 被动 all-in
        return {"player_id": self.id, "action": "call", "amount": amount}

    def raise_bet(self, total_bet_amount: int) -> dict:
        """
        加注。
        total_bet_amount 是「本街道的目标总下注额」，
        例如当前下注 50，我要加注到 150，total_bet_amount=150。
        实际付出 = 目标 - 已下注，同样用 min() 防止超出筹码。
        """
        amount = min(total_bet_amount - self.current_street_bet, self.chips)
        self.chips -= amount
        self.current_street_bet += amount
        self.total_bet += amount
        if self.chips == 0:
            self.is_all_in = True
        # 注意：返回的 amount 是「目标总额」，便于其他玩家知道需要跟注多少
        return {"player_id": self.id, "action": "raise", "amount": total_bet_amount}

    def check(self) -> dict:
        """过牌：无需付出筹码（只在当前 current_bet=0 时合法）。"""
        return {"player_id": self.id, "action": "check", "amount": 0}

    def post_blind(self, amount: int) -> int:
        """
        发送盲注（小盲/大盲）。
        返回实际发送金额（筹码不足时按实际发）。
        这是强制性下注，不算入主动行动次数。
        """
        actual = min(amount, self.chips)
        self.chips -= actual
        self.current_street_bet += actual
        self.total_bet += actual
        if self.chips == 0:
            self.is_all_in = True
        return actual

    def reset_for_new_street(self) -> None:
        """
        新街道开始时重置本街下注额。
        每条街道（flop/turn/river）开始时调用，
        current_street_bet 重置为 0，下注重新从零开始。
        """
        self.current_street_bet = 0

    def reset_for_new_round(self) -> None:
        """
        新一局开始时重置玩家状态。
        chips 保留（跨局累积），其余全部归零。
        """
        self.hole_cards = []
        self.is_active = True
        self.is_all_in = False
        self.current_street_bet = 0
        self.total_bet = 0
        self.thought = ""

    # ── 序列化方法：对象 <-> 字典 互转 ─────────────────────────────────────

    def to_state(self) -> dict:
        """
        序列化为 PlayerState 字典，可直接存入 GameState.players。
        节点函数在操作完 Player 对象后，调用此方法写回 State。
        """
        return {
            "id": self.id,
            "name": self.name,
            "chips": self.chips,
            "hole_cards": self.hole_cards,
            "is_active": self.is_active,
            "is_all_in": self.is_all_in,
            "is_human": self.is_human,
            "is_ai": self.is_ai,
            "current_street_bet": self.current_street_bet,
            "total_bet": self.total_bet,
            "position": self.position,
            "player_type": self.player_type,
            "vpip": self.vpip,
            "pfr": self.pfr,
            "thought": self.thought,
        }

    @classmethod
    def from_state(cls, state: dict) -> "Player":
        """
        从 PlayerState 字典反序列化还原 Player 对象。
        @classmethod 让我们不需要先创建实例就能调用：Player.from_state(d)
        __dataclass_fields__ 是 dataclass 自动生成的字段名集合，
        用它过滤掉 state 里可能存在的多余字段（如 agent_decision 等）。
        """
        return cls(**{k: state[k] for k in cls.__dataclass_fields__ if k in state})


# ── 预设 AI 玩家人设配置表 ────────────────────────────────────────────────────
# 【可扩展点】：想加新人设？在这里加一条，再在 poker_prompts.py 加对应 system prompt

PLAYER_ARCHETYPES: dict[str, dict] = {
    "aggressive": {
        "player_type": "aggressive",
        "description": "激进型：频繁加注和诈唬，喜欢施压对手",
    },
    "passive": {
        "player_type": "passive",
        "description": "保守型：只在强牌时入池，倾向跟注而非加注",
    },
    "balanced": {
        "player_type": "balanced",
        "description": "均衡型：综合运用 GTO 策略，攻守兼备",
    },
    "bluffer": {
        "player_type": "bluffer",
        "description": "诈唬型：极度依赖心理战术，常在弱牌时大额加注",
    },
    "math": {
        "player_type": "math",
        "description": "数学流：严格按底池赔率和期望值决策，不受情绪影响",
    },
}
