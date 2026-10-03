"""测量精度反求（POST /api/v1/precision-budgets）端到端测试。"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import storage


@pytest.fixture(scope="module")
def client():
    storage.init_db()
    with TestClient(app) as c:
        yield c


def _sum_model(version: str, ua: float, ub: float, r: float | None) -> dict:
    """无量纲线性模型 y = a + b（r 为 None 表示不相关）。"""
    return {
        "version": version,
        "name": "线性和 y=a+b（精度反求测试）",
        "expression": "a + b",
        "inputs": [
            {"name": "a", "value": 3.0, "unit": "dimensionless",
             "standard_uncertainty": ua, "distribution": "normal"},
            {"name": "b", "value": 5.0, "unit": "dimensionless",
             "standard_uncertainty": ub, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [] if r is None else [{"i": "a", "j": "b", "r": r}],
    }


def _budget_payload(version: str, target: float = 1.0, unit: str = "dimensionless",
                    input_name: str = "a") -> dict:
    return {
        "model_version": version,
        "input": input_name,
        "target_standard_uncertainty": target,
        "target_unit": unit,
    }


def _register(client: TestClient, model: dict) -> None:
    r = client.post("/api/v1/models", json=model)
    assert r.status_code == 201, r.text


# ---------------------------------------------------------------------------
# 负相关：可行区间有下界（不能把负相关误算成单一上限）
# ---------------------------------------------------------------------------

def test_negative_correlation_interval_has_lower_and_upper(client):
    # y=a+b、r=-1、ub=2、目标 1 -> u_c(ua)=|ua-2| <= 1 -> ua ∈ [1, 3]（两端闭）
    _register(client, _sum_model("budget-rm1", 0.1, 2.0, -1.0))
    r = client.post("/api/v1/precision-budgets", json=_budget_payload("budget-rm1"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_version"] == "budget-rm1"
    assert body["input"] == "a"
    interval = body["interval"]
    assert interval["empty"] is False
    assert interval["unit"] == "dimensionless"
    assert abs(interval["lower"] - 1.0) < 1e-9
    assert abs(interval["upper"] - 3.0) < 1e-9
    assert interval["lower_inclusive"] is True
    assert interval["upper_inclusive"] is True
    # 当前 ua=0.1 -> u_c = |0.1-2| = 1.9 > 1：不达标
    current = body["current"]
    assert abs(current["combined_standard_uncertainty"] - 1.9) < 1e-9
    assert current["meets_target"] is False
    assert current["standard_uncertainty"] == 0.1
    assert body["target"]["standard_uncertainty_in_output_unit"] == 1.0


def test_current_value_inside_interval_meets_target(client):
    # 同一模型，当前 ua=2.0 -> u_c = 0 <= 1：达标，且当前值落在 [1,3] 内
    _register(client, _sum_model("budget-rm1-ok", 2.0, 2.0, -1.0))
    r = client.post("/api/v1/precision-budgets", json=_budget_payload("budget-rm1-ok"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["current"]["meets_target"] is True
    assert body["current"]["combined_standard_uncertainty"] == 0.0
    interval = body["interval"]
    assert interval["lower"] <= 2.0 <= interval["upper"]


def test_zero_target_perfect_cancellation_point(client):
    # 目标 0：u_c(ua)=|ua-2| <= 0 -> 唯一可行点 ua=2，区间退化为 [2, 2]
    r = client.post("/api/v1/precision-budgets",
                    json=_budget_payload("budget-rm1", target=0.0))
    assert r.status_code == 200, r.text
    interval = r.json()["interval"]
    assert interval["empty"] is False
    assert abs(interval["lower"] - 2.0) < 1e-9
    assert abs(interval["upper"] - 2.0) < 1e-9
    assert interval["lower_inclusive"] is True
    assert interval["upper_inclusive"] is True


# ---------------------------------------------------------------------------
# 零端点开闭性：相关输入的零端点开，不相关的零端点闭
# ---------------------------------------------------------------------------

def test_correlated_zero_endpoint_is_open(client):
    # y=a+b、r=+1、ub=0.2、目标 1 -> u_c(ua)=ua+0.2 <= 1 -> ua ∈ [0, 0.8]；
    # 但 a 参与相关，u_a=0 不是合法模型状态 -> 下端点开
    _register(client, _sum_model("budget-rp1", 0.1, 0.2, 1.0))
    r = client.post("/api/v1/precision-budgets", json=_budget_payload("budget-rp1"))
    assert r.status_code == 200, r.text
    interval = r.json()["interval"]
    assert interval["empty"] is False
    assert interval["lower"] == 0.0
    assert interval["lower_inclusive"] is False
    assert abs(interval["upper"] - 0.8) < 1e-9
    assert interval["upper_inclusive"] is True


def test_uncorrelated_zero_endpoint_is_closed(client):
    # 不相关：u_c(ua)=sqrt(ua^2+0.04) <= 1 -> ua ∈ [0, sqrt(0.96)]，下端点闭
    _register(client, _sum_model("budget-unc", 0.1, 0.2, None))
    r = client.post("/api/v1/precision-budgets", json=_budget_payload("budget-unc"))
    assert r.status_code == 200, r.text
    interval = r.json()["interval"]
    assert interval["empty"] is False
    assert interval["lower"] == 0.0
    assert interval["lower_inclusive"] is True
    assert abs(interval["upper"] - math.sqrt(0.96)) < 1e-9
    assert interval["upper_inclusive"] is True


def test_degenerate_zero_interval_of_correlated_input_is_empty(client):
    # y=a+b、r=+1、ub=2、目标 2 -> u_c(ua)=ua+2 <= 2 -> 唯一可行点 ua=0；
    # 相关输入的零端点无效 -> 空区间
    _register(client, _sum_model("budget-degenerate", 0.1, 2.0, 1.0))
    r = client.post("/api/v1/precision-budgets",
                    json=_budget_payload("budget-degenerate", target=2.0))
    assert r.status_code == 200, r.text
    interval = r.json()["interval"]
    assert interval["empty"] is True
    assert interval["lower"] is None and interval["upper"] is None


# ---------------------------------------------------------------------------
# 零灵敏度与无法达标
# ---------------------------------------------------------------------------

def _product_model(version: str) -> dict:
    # y = a*b，b 名义值 0 -> ∂y/∂a = b = 0（a 零灵敏度）；u_c = |a|*u_b = 1.0
    return {
        "version": version,
        "name": "乘积模型 y=a*b（零灵敏度测试）",
        "expression": "a * b",
        "inputs": [
            {"name": "a", "value": 2.0, "unit": "dimensionless",
             "standard_uncertainty": 0.1, "distribution": "normal"},
            {"name": "b", "value": 0.0, "unit": "dimensionless",
             "standard_uncertainty": 0.5, "distribution": "normal"},
        ],
        "output_unit": "dimensionless",
        "correlations": [],
    }


def test_zero_sensitivity_unbounded_interval(client):
    _register(client, _product_model("budget-zero-sens"))
    r = client.post("/api/v1/precision-budgets",
                    json=_budget_payload("budget-zero-sens", target=1.5))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sensitivity"]["coefficient"] == 0.0
    interval = body["interval"]
    assert interval["empty"] is False
    assert interval["lower"] == 0.0
    assert interval["lower_inclusive"] is True
    assert interval["upper"] is None  # 无上限
    assert interval["upper_inclusive"] is False
    # 当前 u_c = 1.0 <= 1.5：达标
    assert body["current"]["meets_target"] is True
    assert abs(body["current"]["combined_standard_uncertainty"] - 1.0) < 1e-9


def test_zero_sensitivity_infeasible_is_empty(client):
    # 同一模型目标 0.5 < u_c = 1.0：无论如何调整 a 都无法达标 -> 空区间
    r = client.post("/api/v1/precision-budgets",
                    json=_budget_payload("budget-zero-sens", target=0.5))
    assert r.status_code == 200, r.text
    body = r.json()
    interval = body["interval"]
    assert interval["empty"] is True
    assert interval["lower"] is None
    assert interval["upper"] is None
    assert body["current"]["meets_target"] is False


def test_infeasible_target_returns_empty_interval(client):
    # 不相关、ub=2、目标 1：u_c^2 = ua^2 + 4 恒 > 1 -> 空区间
    _register(client, _sum_model("budget-infeasible", 0.1, 2.0, None))
    r = client.post("/api/v1/precision-budgets", json=_budget_payload("budget-infeasible"))
    assert r.status_code == 200, r.text
    assert r.json()["interval"]["empty"] is True


# ---------------------------------------------------------------------------
# 单位换算：目标带单位，端点落在输入声明单位下
# ---------------------------------------------------------------------------

def test_target_unit_conversion_and_declared_unit_endpoints(client):
    # P = U*I：U=10(0.1) V、I=2(0.02) A、r=0.5，输出 W。
    # u_c^2(u_U) = (I*u_U)^2 + (U*u_I)^2 + 2*I*U*r*u_U*u_I
    #            = 4 u_U^2 + 0.4 u_U + 0.04  （u_U 以 V 计）
    # 目标 300 mW = 0.3 W -> 4x^2+0.4x+0.04 <= 0.09 -> x <= (-0.4+sqrt(0.96))/8
    model = {
        "version": "budget-power",
        "expression": "U * I",
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1,
             "distribution": "normal"},
            {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.02,
             "distribution": "normal"},
        ],
        "output_unit": "W",
        "correlations": [{"i": "U", "j": "I", "r": 0.5}],
    }
    _register(client, model)
    r = client.post("/api/v1/precision-budgets", json={
        "model_version": "budget-power",
        "input": "U",
        "target_standard_uncertainty": 300.0,
        "target_unit": "mW",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target"]["standard_uncertainty_in_output_unit"] == 0.3
    interval = body["interval"]
    expected_upper = (-0.4 + math.sqrt(0.96)) / 8.0
    assert interval["unit"] == "V"
    assert interval["empty"] is False
    assert interval["lower"] == 0.0
    assert interval["lower_inclusive"] is False  # U 参与相关，零端点开
    assert abs(interval["upper"] - expected_upper) < 1e-9
    assert interval["upper_inclusive"] is True
    # 当前 u_c = sqrt(0.12) ≈ 0.3464 W > 0.3 W：不达标
    assert abs(body["current"]["combined_standard_uncertainty"] - math.sqrt(0.12)) < 1e-9
    assert body["current"]["meets_target"] is False


# ---------------------------------------------------------------------------
# 拒绝场景
# ---------------------------------------------------------------------------

def test_unknown_model_version_rejected(client):
    r = client.post("/api/v1/precision-budgets", json=_budget_payload("no-such-version"))
    assert r.status_code == 404


def test_unknown_input_rejected(client):
    r = client.post("/api/v1/precision-budgets",
                    json=_budget_payload("budget-rm1", input_name="zz"))
    assert r.status_code == 422
    assert "zz" in r.json()["message"]


def test_target_dimension_mismatch_rejected(client):
    r = client.post("/api/v1/precision-budgets",
                    json=_budget_payload("budget-rm1", unit="meter"))
    assert r.status_code == 422
    assert "量纲" in r.json()["message"]


def test_unknown_target_unit_rejected(client):
    r = client.post("/api/v1/precision-budgets",
                    json=_budget_payload("budget-rm1", unit="not-a-unit"))
    assert r.status_code == 422


def test_negative_target_rejected(client):
    r = client.post("/api/v1/precision-budgets",
                    json=_budget_payload("budget-rm1", target=-0.5))
    assert r.status_code == 422


def test_non_finite_target_rejected(client):
    # httpx 的 json= 拒绝非有限浮点，改用原始 JSON 文本（NaN/Infinity 字面量）
    import json as jsonlib

    for bad in (float("nan"), float("inf"), float("-inf")):
        payload = jsonlib.dumps(_budget_payload("budget-rm1", target=bad))
        r = client.post("/api/v1/precision-budgets", content=payload,
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 422, bad


# ---------------------------------------------------------------------------
# 只读语义：不改模型、不写计算记录，原有传播/幂等/查询不受影响
# ---------------------------------------------------------------------------

def test_inverse_is_read_only_and_propagation_still_works(client):
    model = _sum_model("budget-readonly", 0.1, 0.2, 0.5)
    _register(client, model)
    before_model = client.get("/api/v1/models/budget-readonly").json()
    before_evals = client.get("/api/v1/evaluations?model_version=budget-readonly").json()

    r = client.post("/api/v1/precision-budgets", json=_budget_payload("budget-readonly"))
    assert r.status_code == 200, r.text

    # 模型未被修改，且没有写入任何计算记录
    assert client.get("/api/v1/models/budget-readonly").json() == before_model
    after_evals = client.get("/api/v1/evaluations?model_version=budget-readonly").json()
    assert after_evals["count"] == before_evals["count"] == 0

    # 原有传播、幂等与查询仍有效
    payload = {
        "model_version": "budget-readonly",
        "inputs": model["inputs"],
        "correlations": model["correlations"],
        "n": 5000,
        "seed": 7,
        "idempotency_key": "key-budget-readonly-0001",
    }
    r1 = client.post("/api/v1/evaluations", json=payload)
    assert r1.status_code == 201, r1.text
    r2 = client.post("/api/v1/evaluations", json=payload)
    assert r2.status_code == 200 and r2.json()["cached"] is True
    assert r2.json()["id"] == r1.json()["id"]
    assert client.get(f"/api/v1/evaluations/{r1.json()['id']}").status_code == 200
    assert client.get("/api/v1/idempotency/key-budget-readonly-0001").status_code == 200
