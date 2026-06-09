from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_job_monitor_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_telegram_job_monitor.py"
    spec = importlib.util.spec_from_file_location("run_telegram_job_monitor", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_product_manager_title_skips_even_with_data_analysis_responsibilities() -> None:
    mod = _load_job_monitor_module()
    text = """待招岗位：高级产品经理 / 产品
薪酬福利：面议
岗位职责：负责用户交易行为数据分析，制定增长策略和功能规划
岗位要求：熟悉 Web3 和交易产品"""

    result = mod.evaluate_job_message(text)

    assert not result.matched
    assert result.skipped_reason == "skipped_irrelevant_role"
    assert "产品经理" in result.matched_irrelevant_role_keywords
    assert "数据分析" in result.matched_responsibility_keywords


def test_parse_job_sections_extracts_hashtag_title() -> None:
    mod = _load_job_monitor_module()

    sections = mod.parse_job_sections("📚 待招岗位：#量化研究员 #研究")

    assert sections.title_or_roles == "量化研究员 研究"


def test_parse_job_sections_extracts_dejob_emoji_sections() -> None:
    mod = _load_job_monitor_module()
    text = """📚 待招岗位：#量化研究员 #研究
💰 薪酬福利：$2500 - $5000 / month
🌱 岗位职责：
1️⃣ 负责做市、交易机制、风险控制和流动性建设
🌵 岗位要求：
1️⃣ 熟悉 Perp DEX / CEX 交易机制
📮 申请方式：联系 HR"""

    sections = mod.parse_job_sections(text)

    assert sections.title_or_roles == "量化研究员 研究"
    assert "做市" in sections.responsibilities
    assert "风险控制" in sections.responsibilities
    assert "Perp DEX" in sections.requirements
    assert "CEX" in sections.requirements


def test_parse_job_sections_extracts_apply_method_and_source() -> None:
    mod = _load_job_monitor_module()
    text = """📚 待招岗位：#量化研究员 #研究
📮 申请方式：
Email: jobs@example.com
请附上简历和过往策略案例
🔗 岗位来源：https://www.dejob.ai/jobDetail?id=6990
官网：https://gperp.example"""

    sections = mod.parse_job_sections(text)

    assert "Email: jobs@example.com" in sections.apply_method
    assert "过往策略案例" in sections.apply_method
    assert sections.source == "https://www.dejob.ai/jobDetail?id=6990"
    assert "官网: https://gperp.example" in sections.official_links


def test_build_telegram_message_link_for_private_channel_id() -> None:
    mod = _load_job_monitor_module()
    chat = SimpleNamespace(id=-1001570628112, username=None)
    message = SimpleNamespace(id=8207)

    assert mod.build_telegram_message_link(chat, message) == "https://t.me/c/1570628112/8207"


def test_build_telegram_message_link_for_public_username() -> None:
    mod = _load_job_monitor_module()
    chat = SimpleNamespace(id=-1001570628112, username="dejob_channel")
    message = SimpleNamespace(id=8207)

    assert mod.build_telegram_message_link(chat, message) == "https://t.me/dejob_channel/8207"


def test_gperp_quant_researcher_title_matches() -> None:
    mod = _load_job_monitor_module()
    text = """Gperp / #DEX
合作方式：兼职 / 远程 / 实地
待招岗位：量化研究员 / 研究
薪酬福利：$2500 - $5000 / month，代币分红
岗位职责：交易机制、流动性建设、做市体系、风险控制、市场风险、交易深度、资金费率等
岗位要求：有相关经验等"""

    result = mod.evaluate_job_message(text)

    assert result.matched
    assert result.sections.title_or_roles == "量化研究员 / 研究"
    assert "量化研究员" in result.matched_title_keywords


def test_gperp_dejob_emoji_sample_matches() -> None:
    mod = _load_job_monitor_module()
    text = """Gperp / #DEX
📚 待招岗位：#量化研究员 #研究
💰 薪酬福利：$2500 - $5000 / month
🌱 岗位职责：
1️⃣ 负责交易机制、流动性建设、做市体系、风险控制
🌵 岗位要求：
1️⃣ 熟悉 Perp DEX / CEX 交易机制
📮 申请方式：联系招聘方"""

    result = mod.evaluate_job_message(text)

    assert result.matched
    assert result.sections.title_or_roles == "量化研究员 研究"
    assert "量化研究员" in result.matched_title_keywords
    assert "做市" in result.matched_responsibility_keywords


def test_gperp_notification_includes_apply_source_and_message_link() -> None:
    mod = _load_job_monitor_module()
    text = """Gperp / #DEX
📚 待招岗位：#量化研究员 #研究
🤝 合作方式：兼职 / 远程 / 实地
💰 薪酬福利：$2500 - $5000 / month，代币分红
🌱 岗位职责：
1️⃣ 负责交易机制、流动性建设、做市体系、风险控制
🌵 岗位要求：
1️⃣ 熟悉 Perp DEX / CEX 交易机制
📮 申请方式：
Email: jobs@example.com
🔗 岗位来源：https://www.dejob.ai/jobDetail?id=6990"""
    message = SimpleNamespace(id=8207)

    rendered = mod._format_notify_message(
        [
            (
                message,
                ["量化研究员", "做市"],
                text,
                "https://t.me/c/1570628112/8207",
            )
        ]
    )

    assert "合作方式：兼职 / 远程 / 实地" in rendered
    assert "薪酬福利：$2500 - $5000 / month，代币分红" in rendered
    assert "岗位职责：" in rendered
    assert "1. 负责交易机制、流动性建设、做市体系、风险控制" in rendered
    assert "岗位要求：" in rendered
    assert "1. 熟悉 Perp DEX / CEX 交易机制" in rendered
    assert "申请方式：\nEmail: jobs@example.com" in rendered
    assert "岗位来源：\nhttps://www.dejob.ai/jobDetail?id=6990" in rendered
    assert "原消息：\nhttps://t.me/c/1570628112/8207" in rendered


def test_long_notification_splits_without_dropping_content() -> None:
    mod = _load_job_monitor_module()
    body = "\n".join(f"line-{index} " + ("x" * 80) for index in range(120))
    chunks = mod._split_for_telegram(body, limit=3500)

    assert len(chunks) > 1
    assert chunks[0].startswith(f"第 1/{len(chunks)} 部分")
    assert chunks[-1].startswith(f"第 {len(chunks)}/{len(chunks)} 部分")
    joined = "\n".join(
        chunk.split("\n\n", 1)[1] if chunk.startswith("第 ") else chunk
        for chunk in chunks
    )
    assert "line-0" in joined
    assert "line-119" in joined


def test_market_making_quant_title_matches() -> None:
    mod = _load_job_monitor_module()
    result = mod.evaluate_job_message("待招岗位：做市研究员 / 量化\n岗位职责：负责策略研究")

    assert result.matched
    assert "做市" in result.matched_title_keywords
    assert "量化" in result.matched_title_keywords


def test_ai_agent_intern_title_matches() -> None:
    mod = _load_job_monitor_module()
    result = mod.evaluate_job_message("待招岗位：AI Agent 实习生\n岗位职责：负责工具调用工作流")

    assert result.matched
    assert "AI Agent" in result.matched_title_keywords


def test_data_analyst_title_matches() -> None:
    mod = _load_job_monitor_module()
    result = mod.evaluate_job_message("待招岗位：数据分析师\n岗位职责：负责业务指标分析")

    assert result.matched
    assert "数据分析师" in result.matched_title_keywords


def test_responsibilities_data_analysis_does_not_override_product_title() -> None:
    mod = _load_job_monitor_module()
    text = """招聘岗位：产品经理
岗位职责：负责数据分析、用户研究、交易行为洞察
岗位要求：熟悉增长模型"""

    result = mod.evaluate_job_message(text)

    assert not result.matched
    assert result.skipped_reason == "skipped_irrelevant_role"
    assert "产品经理" in result.matched_irrelevant_role_keywords


def test_bounty_referral_ad_is_excluded() -> None:
    mod = _load_job_monitor_module()
    text = "Referral Bounty：推荐奖励和中介佣金，关注频道领取机会市场资源。"

    result = mod.evaluate_job_message(text)

    assert not result.matched
    assert result.skipped_reason == "skipped_by_exclude_keywords"
    assert "Referral" in result.matched_exclude_keywords
    assert "Bounty" in result.matched_exclude_keywords


def test_fallback_requires_two_targets_or_one_strong_target() -> None:
    mod = _load_job_monitor_module()
    one_weak = """招聘
岗位职责：负责数据分析和报表
岗位要求：沟通能力强"""
    two_targets = """招聘
岗位职责：负责数据分析和因子分析
岗位要求：熟悉回测"""
    strong_target = """招聘
岗位职责：负责 Market Maker 系统
岗位要求：有交易经验"""

    assert not mod.evaluate_job_message(one_weak).matched
    assert mod.evaluate_job_message(two_targets).matched
    assert mod.evaluate_job_message(strong_target).matched
