# -*- coding: utf-8 -*-
"""
TuShare：港股 + A股 抓取（日终收盘 + 近20周周线）+ 一键输入“代码@成本”
改动要点（只用 TuShare）：
- 每只股票只调用 1 次：港股用 hk_daily；A股用 daily
- 最新收盘价 = 该日线结果的最后一条 close
- 近20周周线 = 该日线本地 resample("W-FRI")
- 自动识别代码：5位纯数字→港股；6位纯数字→A股(.SH/.SZ按前缀推断)；也可直接输入带后缀
- 成本只从 INPUTS 里解析（“代码@成本”），不再交互输入
"""

import os, time, re
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
import requests
import random


# ====== Server酱配置（方糖服务号）======
# 必须使用环境变量注入，否则报错
SERVERCHAN_SENDKEY = os.environ.get("SERVERCHAN_SENDKEY")

if not SERVERCHAN_SENDKEY:
    raise ValueError("未配置 Server酱 SendKey（环境变量 SERVERCHAN_SENDKEY）")

def push_serverchan(title: str, content: str):
    """
    使用 Server酱 Turbo 把文本推到微信（方糖服务号）
    """
    url = f"https://sctapi.ftqq.com/{SERVERCHAN_SENDKEY}.send"
    data = {"title": title, "desp": content}

    try:
        resp = requests.post(url, data=data, timeout=10)
        print("Server酱推送结果：", resp.text)
    except Exception as e:
        print("Server酱推送失败：", e)



# ========== 配置区：必须使用环境变量 ==========
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN")
if not TUSHARE_TOKEN:
    raise ValueError("未配置 TuShare Token（环境变量 TUSHARE_TOKEN）")

pro = ts.pro_api(TUSHARE_TOKEN)

SLEEP_SEC = random.uniform(32, 38)





# 在这里填写你的输入（任选一种写法）
# 既可混合港股与A股，也可带成本，如 ["00700.HK@320", "600519.SH@1650", "000001.SZ"]
INPUTS = [
    "01919.HK@13.33",
    "00883.HK@21.96",
    "01088.HK@40.42",
    "01898.HK@10.86",
    "00700.HK@621",
    "00941.HK@86.5",
    "600900.SH@28.68",
]

# ========== 初始化 TuShare ==========
if not TUSHARE_TOKEN or TUSHARE_TOKEN == "YOUR_TUSHARE_TOKEN":
    raise ValueError("请设置 TuShare Token（环境变量 TUSHARE_TOKEN，或直接替换代码中的占位符）。")
pro = ts.pro_api(TUSHARE_TOKEN)


# ========== 工具：代码规范化 & 输入解析 ==========
def _infer_cn_suffix(code6: str) -> str:
    """6位数字推断A股后缀：上交所(.SH) or 深交所(.SZ)"""
    assert code6.isdigit() and len(code6) == 6
    # 常见规则：6/60/61/603/605/68 科创板 → .SH； 000/001/002/003/300/301/302 → .SZ
    if code6.startswith(("60", "61", "68", "603", "605")) or code6[0] == "6":
        return code6 + ".SH"
    else:
        return code6 + ".SZ"


def normalize_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if s.endswith((".HK", ".SH", ".SZ")):
        core = s[:-3]
        if core.isdigit():
            if s.endswith(".HK"):
                return core.zfill(5) + ".HK"  # 港股5位
            else:
                return core.zfill(6) + s[-3:]  # A股6位
        return s
    # 纯数字自动推断
    if s.isdigit():
        if len(s) == 5:
            return s.zfill(5) + ".HK"
        elif len(s) == 6:
            return _infer_cn_suffix(s.zfill(6))
    # 形如 "00700"（5位）也会识别成港股
    if re.fullmatch(r"\d{5}", s):
        return s + ".HK"
    if re.fullmatch(r"\d{6}", s):
        return _infer_cn_suffix(s)
    return s  # 其它原样放过


def parse_inputs(inputs: Any) -> Tuple[List[str], Dict[str, float]]:
    """
    解析 INPUTS：
    - 支持 "代码" / "代码@成本" / [代码, 成本] / dict 形式
    - 返回：标准化后的代码列表 + {代码: 成本}
    """
    symbols: List[str] = []
    costs: Dict[str, float] = {}

    if isinstance(inputs, dict):
        for k, v in inputs.items():
            sym = normalize_symbol(str(k))
            if sym not in symbols:
                symbols.append(sym)
            try:
                costs[sym] = float(v)
            except Exception:
                pass

    elif isinstance(inputs, (list, tuple)):
        for item in inputs:
            if isinstance(item, str) and "@" in item:
                code, c = item.split("@", 1)
                sym = normalize_symbol(code)
                if sym not in symbols:
                    symbols.append(sym)
                try:
                    costs[sym] = float(c)
                except Exception:
                    pass
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                sym = normalize_symbol(str(item[0]))
                if sym not in symbols:
                    symbols.append(sym)
                if len(item) >= 2:
                    try:
                        costs[sym] = float(item[1])
                    except Exception:
                        pass
            elif isinstance(item, str):
                sym = normalize_symbol(item)
                if sym not in symbols:
                    symbols.append(sym)
    else:
        raise ValueError("INPUTS 类型不支持，请使用 dict / list / tuple / '代码@成本' 等形式。")

    # 只保留标准 ts_code
    symbols = [s for s in symbols if s.endswith((".HK", ".SH", ".SZ"))]
    return symbols, costs


SYMBOLS, COST_BASIS = parse_inputs(INPUTS)


# ========== TuShare 抓取（每股只调 1 次，自动路由） ==========
def _fetch_daily_any(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    自动路由：
      - 港股：pro.hk_daily
      - A股： pro.daily
    返回统一列：trade_date, open, high, low, close, vol, amount
    """
    if ts_code.endswith(".HK"):
        df = pro.hk_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    else:
        # A股
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    if df is None or df.empty:
        return pd.DataFrame()

    # 统一整理
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "vol", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.NA  # 缺哪个列就补哪个列，保证统一
    return df[["trade_date", "open", "high", "low", "close", "vol", "amount"]]


def get_weekly20_and_last_close(symbol: str, back_weeks: int = 160):
    """
    一次接口（日线）拿齐：
      - 最新收盘价 = 日线最后一条 close
      - 近20周周线 = 日线重采样 W-FRI 的最后20根
    """
    ts_code = normalize_symbol(symbol)
    end_date = datetime.today().strftime("%Y%m%d")
    start_date = (datetime.today() - timedelta(weeks=back_weeks)).strftime("%Y%m%d")
    daily = _fetch_daily_any(ts_code, start_date, end_date)
    if daily.empty:
        return (pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"]), None, None)

    # 最新收盘（日终）
    last_close = float(daily["close"].iloc[-1])
    last_trade_date = daily["trade_date"].iloc[-1].date()

    # 本地重采样成周线（以周五为周线结束日）
    idx = daily.set_index("trade_date")
    wk = pd.DataFrame({
        "open":   idx["open"].resample("W-FRI").first(),
        "high":   idx["high"].resample("W-FRI").max(),
        "low":    idx["low"].resample("W-FRI").min(),
        "close":  idx["close"].resample("W-FRI").last(),
        "volume": idx["vol"].resample("W-FRI").sum(min_count=1),
    }).dropna(subset=["open", "high", "low", "close"], how="any").reset_index().rename(columns={"trade_date": "date"})
    wk.insert(0, "symbol", ts_code)
    weekly_20 = wk.tail(20).reset_index(drop=True)

    return weekly_20, last_close, last_trade_date


def fetch_all_symbols(symbols_list: List[str]):
    """
    批量抓取（每股只调 1 次）：
      - weekly_dict: {symbol -> 近20周周线 DataFrame}
      - latest_df:   DataFrame(symbol, price, trade_date)
      - wide_df:     周收盘宽表
    """
    weekly_dict, latest_rows = {}, []
    for sym in symbols_list:
        sym = normalize_symbol(sym)
        print(f"抓取（周线20 + 最新收盘）：{sym}")
        wk20, last_px, last_dt = get_weekly20_and_last_close(sym)
        weekly_dict[sym] = wk20
        latest_rows.append({"symbol": sym, "price": last_px, "trade_date": last_dt})
        time.sleep(SLEEP_SEC)  # 温和节流

    latest_df = pd.DataFrame(latest_rows, columns=["symbol", "price", "trade_date"])

    parts = []
    for sym, df in weekly_dict.items():
        if df is not None and not df.empty:
            parts.append(df.set_index("date")["close"].rename(sym))
    wide_df = pd.concat(parts, axis=1).sort_index() if parts else pd.DataFrame()
    return weekly_dict, latest_df, wide_df


# ========== 分析函数 ==========
def _safe_last_ma(series: pd.Series, window: int):
    if series is None or series.empty or len(series) < window:
        return None
    return series.rolling(window).mean().iloc[-1]


def classify_trend(close_series: pd.Series, current_price: float) -> dict:
    s = close_series.dropna().astype(float)
    ma4 = _safe_last_ma(s, 4)
    ma8 = _safe_last_ma(s, 8)
    ma16 = _safe_last_ma(s, 16)
    if ma4 is None or ma8 is None or ma16 is None:
        return {"status": "数据不足", "ma4": ma4, "ma8": ma8, "ma16": ma16}
    if ma4 > ma8 > ma16 and current_price > ma4:
        status = "强势上升"
    elif ma4 < ma8 < ma16 and current_price < ma4:
        status = "强势下降"
    else:
        spread = (max(ma4, ma8, ma16) - min(ma4, ma8, ma16)) / ma8 if ma8 else 1.0
        near_ma8 = abs(current_price - ma8) / ma8 if ma8 else 1.0
        if spread < 0.05 and near_ma8 <= 0.08:
            status = "横盘震荡"
        else:
            status = "弱势上升" if ma4 > ma8 else "弱势下降"
    return {"status": status, "ma4": ma4, "ma8": ma8, "ma16": ma16}


def price_position(close_series: pd.Series, current_price: float, lookback: int = 16) -> dict:
    s = close_series.dropna().astype(float).tail(lookback)
    if s.empty:
        return {"pos_pct": None, "level": "数据不足", "hi": None, "lo": None}
    hi, lo = float(s.max()), float(s.min())
    rng = hi - lo
    if hi <= 0 or rng <= 0:
        return {"pos_pct": None, "level": "数据不足", "hi": hi, "lo": lo}
    pos = (current_price - lo) / rng
    level = "高位" if pos > 0.70 else ("低位" if pos < 0.30 else "中位")
    return {"pos_pct": float(pos), "level": level, "hi": hi, "lo": lo}


def peak_drawdown_buy_signal(close_series: pd.Series, current_price: float) -> dict:
    s = close_series.dropna().astype(float).tail(16)
    if s.empty:
        return {"score": 0, "buy": False, "drawdown": None, "details": {}}
    peak = float(s.max())
    dd = (peak - current_price) / peak if peak > 0 else None
    ma8 = _safe_last_ma(close_series, 8)
    ma16 = _safe_last_ma(close_series, 16)
    trend = classify_trend(close_series, current_price)["status"]
    score, details = 0, {}
    if dd is not None:
        if dd >= 0.15:
            score += 3
            details["dd>=15%"] = 3
        if dd >= 0.25:
            score += 2
            details["dd>=25%"] = 2
    if ma16 is not None and current_price < ma16:
        score += 2
        details["px<MA16"] = 2
    if ma8 is not None and current_price < ma8:
        score += 1
        details["px<MA8"] = 1
    if trend in ("强势下降", "弱势下降"):
        score += 2
        details["downtrend"] = 2
    return {"score": score, "buy": score >= 5, "drawdown": dd, "details": details}


def volatility_20w(close_series: pd.Series) -> Optional[float]:
    rets = close_series.dropna().astype(float).pct_change().dropna().tail(20)
    if rets.empty:
        return None
    return float(rets.std() * 100.0)


def dynamic_take_profit(cost: Optional[float], current_price: float, close_series: pd.Series, trend_status: str) -> dict:
    if cost is None or cost <= 0:
        return {"available": False, "reason": "未提供成本价"}
    vol_pct = volatility_20w(close_series)
    if vol_pct is not None and vol_pct > 8.0:
        vol_factor = 1.3
    elif vol_pct is not None and vol_pct < 4.0:
        vol_factor = 0.8
    else:
        vol_factor = 1.0
    if trend_status == "强势上升":
        trend_factor = 1.2
    elif trend_status in ("强势下降", "弱势下降"):
        trend_factor = 0.8
    else:
        trend_factor = 1.0
    base_thr, base_sell = [0.20, 0.35, 0.50], [0.10, 0.15, 0.25]
    final_thr = [round(t * vol_factor * trend_factor, 4) for t in base_thr]
    pr = (current_price - cost) / cost
    actions = []
    total = 0.0
    for t, s in zip(final_thr, base_sell):
        if pr >= t:
            actions.append({"threshold": t, "sell_ratio": s})
            total += s
    return {
        "available": True,
        "vol_pct": vol_pct,
        "vol_factor": vol_factor,
        "trend_factor": trend_factor,
        "final_thresholds": final_thr,
        "profit_ratio": pr,
        "plan": actions,
        "total_sell_ratio": total,
    }


def averaging_recommendation(cost: Optional[float], current_price: float,
                             weekly_close: pd.Series, weekly_volume: pd.Series,
                             support_near_pct: float = 0.02) -> dict:
    if cost is None or cost <= 0:
        return {"available": False, "reason": "未提供成本价"}
    loss = (cost - current_price) / cost
    s_close = weekly_close.dropna().astype(float)
    s_vol = weekly_volume.dropna().astype(float)
    if s_close.empty or s_vol.empty:
        return {"available": False, "reason": "数据不足"}
    lo8 = float(s_close.tail(8).min())
    lo16 = float(s_close.tail(16).min())
    cur_vol = float(s_vol.iloc[-1]) if len(s_vol) else None
    avg8_vol = float(s_vol.tail(8).mean()) if len(s_vol) else None
    vol_ratio = (cur_vol / avg8_vol) if (cur_vol is not None and avg8_vol and avg8_vol > 0) else None

    def near(px, sup, tol):
        return False if not sup or sup <= 0 else abs(px - sup) / sup <= tol

    near8 = near(current_price, lo8, support_near_pct)
    near16 = near(current_price, lo16, support_near_pct)
    decision = {"action": "不加仓", "size": 0.0}
    if loss >= 0.25:
        decision = {"action": "重新评估投资逻辑", "size": 0.0}
    elif loss >= 0.15 and (near16 or (vol_ratio is not None and vol_ratio < 0.5)):
        decision = {"action": "加仓", "size": 0.30}
    elif loss >= 0.08 and (near8 or (vol_ratio is not None and vol_ratio < 0.7)):
        decision = {"action": "加仓", "size": 0.20}
    return {
        "available": True,
        "loss_ratio": float(loss),
        "support8": lo8,
        "support16": lo16,
        "near8": near8,
        "near16": near16,
        "volume_ratio": vol_ratio,
        "decision": decision,
    }


def risk_control(weekly_close: pd.Series, cost: Optional[float],
                 current_price: float, trend_status: str) -> dict:
    s20 = weekly_close.dropna().astype(float).tail(20)
    if s20.empty:
        return {"available": False, "reason": "数据不足"}
    hi, lo = float(s20.max()), float(s20.min())
    mdd = (hi - lo) / hi if hi > 0 else None
    if mdd is None:
        return {"available": False, "reason": "数据不足"}
    if mdd > 0.35:
        risk, pos = "高风险", "10%-15%"
    elif mdd >= 0.20:
        risk, pos = "中风险", "15%-20%"
    else:
        risk, pos = "低风险", "20%-25%"
    stop, warn = None, None
    if cost is not None and cost > 0:
        loss = (cost - current_price) / cost
        if loss >= 0.30:
            stop = "浮亏≥30%，强制止损"
        elif loss >= 0.20 and trend_status in ("强势下降", "弱势下降"):
            warn = "浮亏≥20%且下降趋势，风险警告"
    return {
        "available": True,
        "mdd": float(mdd),
        "risk_level": risk,
        "position_suggestion": pos,
        "stop_loss": stop,
        "warning": warn,
    }


def analyze_symbol(symbol: str, weekly_df: pd.DataFrame,
                   latest_row: pd.Series, cost: Optional[float] = None) -> dict:
    close = weekly_df["close"]
    volume = weekly_df["volume"] if "volume" in weekly_df.columns else pd.Series([], dtype=float)
    current_price = float(latest_row.get("price", close.iloc[-1]))
    trend = classify_trend(close, current_price)
    pos = price_position(close, current_price, lookback=16)
    buy_sig = peak_drawdown_buy_signal(close, current_price)
    take_profit = dynamic_take_profit(cost, current_price, close, trend["status"])
    avg_rec = averaging_recommendation(cost, current_price, close, volume)
    risk = risk_control(close, cost, current_price, trend["status"])
    return {
        "symbol": symbol,
        "current_price": current_price,
        "trend": trend,
        "position": pos,
        "buy_signal": buy_sig,
        "take_profit": take_profit,
        "averaging": avg_rec,
        "risk": risk,
        "cost": cost,
    }


def summarize_result(res: dict) -> str:
    """
    汇总成一行中文摘要：
    代码：趋势（在区间中的位置），4个月回撤X%，[是否买入信号]，现价/成本/盈亏，浮盈浮亏提示，加仓建议，风险提示
    """
    sym = res["symbol"]
    trend = res["trend"]["status"]
    pos = res["position"]
    pos_str = "未知" if pos.get("pos_pct") is None else f"{pos['level']}({round(pos['pos_pct'] * 100, 1)}%)"
    dd = res["buy_signal"].get("drawdown")
    dd_str = "-" if dd is None else f"{round(dd * 100, 1)}%"
    buy_phrase = " → 关注买入" if res["buy_signal"].get("buy") else ""

    # 成本 + 盈亏
    cost = res.get("cost")
    profit_phrase = ""
    if cost is not None and cost > 0:
        diff = res["current_price"] - cost
        pct = diff / cost
        profit_phrase = f"，现价{res['current_price']:.2f}，成本{cost:.2f}，盈亏{diff:+.2f}({pct * 100:+.1f}%)"

    # 止盈逻辑的浮盈/浮亏描述（可选）
    tp = res["take_profit"]
    take_profit_phrase = ""
    if tp.get("available") and tp.get("profit_ratio") is not None:
        pr = tp["profit_ratio"]
        pr_pct = round(abs(pr) * 100, 1)
        take_profit_phrase = f"，浮盈{pr_pct}%" if pr >= 0 else f"，浮亏{pr_pct}%"

    # 加仓建议
    avg = res["averaging"]
    avg_phrase = ""
    if avg.get("available"):
        loss = avg.get("loss_ratio")
        if loss is not None:
            loss_pct = round(abs(loss) * 100, 1)
            if avg["decision"]["action"] == "加仓":
                size_pct = int(avg["decision"]["size"] * 100)
                avg_phrase = f"，浮亏{loss_pct}%→建议加仓{size_pct}%"
            elif avg["decision"]["action"] == "重新评估投资逻辑":
                avg_phrase = f"，浮亏{loss_pct}%→建议重新评估投资逻辑"

    # 风险提示
    risk = res["risk"]
    risk_phrase = ""
    if risk.get("available"):
        risk_phrase = f"，{risk['risk_level']}→仓位建议{res['risk']['position_suggestion']}"

    return f"{sym}：{trend}（{pos_str}），4个月回撤{dd_str}{buy_phrase}{profit_phrase}{take_profit_phrase}{avg_phrase}{risk_phrase}"


# ========== 实际运行 + 推送到 Server酱 ==========
if __name__ == "__main__":
    # 1）抓数据
    weekly_dict, latest_df, wide_df = fetch_all_symbols(SYMBOLS)

    print("\n=== 输入解析 ===")
    print("SYMBOLS:", SYMBOLS)
    print("COST_BASIS:", COST_BASIS)

    # 把成本和盈亏也打印一下，方便在终端查看
    if not latest_df.empty:
        latest_df["cost"] = latest_df["symbol"].map(COST_BASIS)
        latest_df["pnl"] = latest_df["price"] - latest_df["cost"]
        latest_df["pnl_pct"] = (latest_df["pnl"] / latest_df["cost"] * 100).round(2)
    print("\n=== 最近交易日收盘（来自本次日线的最后一条）+ 成本与盈亏 ===")
    print(latest_df)

    # 随便预览一只票的周线（这里用第一个符号）
    if SYMBOLS:
        sample_sym = SYMBOLS[0]
        print(f"\n=== {sample_sym} 近20周周线（预览） ===")
        print(weekly_dict.get(sample_sym, pd.DataFrame()).tail(5))

    # 把最新价做成 map，便于后面查
    latest_map = {row["symbol"]: row for _, row in latest_df.iterrows()} if not latest_df.empty else {}

    print("\n=== 监测结果（摘要） ===")
    summary_lines: List[str] = []  # 收集每一只股票的文字摘要，用来推送

    for sym, wdf in weekly_dict.items():
        if wdf is None or wdf.empty:
            line = f"{sym}：无周线数据（可能未覆盖/权限不足/当日限额已用尽）"
            print(line)
            summary_lines.append(line)
            continue

        lrow = latest_map.get(sym, pd.Series({}))
        cost = COST_BASIS.get(sym)  # 成本只来自 INPUTS
        res = analyze_symbol(sym, wdf, lrow, cost)
        line = summarize_result(res)
        print(line)
        summary_lines.append(line)

    # 2）整理并推送到微信
    if summary_lines:
        # 标题：带个时间更直观
        title = time.strftime("港股/A股持仓监测 %Y-%m-%d %H:%M", time.localtime())
        # 正文：每只股票一行，中间空一行
        body = "\n\n".join(summary_lines)
        push_serverchan(title, body)
    else:
        print("没有可推送的监测结果，不推送。")

