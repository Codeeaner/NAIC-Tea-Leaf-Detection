"""Deterministic decision-support logic for tea leaf batches.

This service converts detection totals into farmer-facing actions:
sell, process, or discard. It also estimates value at risk and value
recovery using transparent heuristics so the result can be explained
to non-technical users.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


class DecisionSupportService:
    """Build farmer-facing batch recommendations from detection totals."""

    def __init__(self) -> None:
        self.currency = self._normalize_currency(os.getenv("TEALEAF_CURRENCY", "USD"))
        self.value_per_leaf = self._safe_float(os.getenv("TEALEAF_VALUE_PER_LEAF"), 0.05)
        self.assumed_leaves_per_kg = self._safe_float(os.getenv("TEALEAF_ASSUMED_LEAVES_PER_KG"), 3000.0)
        self.minimum_sample_weight_kg = self._safe_float(os.getenv("TEALEAF_MIN_SAMPLE_WEIGHT_KG"), 0.25)
        self.premium_price_per_kg = self._safe_float(os.getenv("TEALEAF_PREMIUM_PRICE_PER_KG"), 8.0)
        self.standard_price_per_kg = self._safe_float(os.getenv("TEALEAF_STANDARD_PRICE_PER_KG"), 5.5)
        self.reject_price_per_kg = self._safe_float(os.getenv("TEALEAF_REJECT_PRICE_PER_KG"), 1.0)
        self.sorting_cost_per_kg = self._safe_float(os.getenv("TEALEAF_SORTING_COST_PER_KG"), 0.4)
        self.rework_cost_per_kg = self._safe_float(os.getenv("TEALEAF_REWORK_COST_PER_KG"), 0.25)
        self.disposal_cost_per_kg = self._safe_float(os.getenv("TEALEAF_DISPOSAL_COST_PER_KG"), 0.1)
        self.default_price_per_kg = self._safe_float(
            os.getenv("TEALEAF_DEFAULT_PRICE_PER_KG"),
            self.premium_price_per_kg,
        )

    def build_decision_support(
        self,
        *,
        session_id: int,
        session_name: str,
        total_images: int,
        completed_images: int,
        healthy_count: int,
        unhealthy_count: int,
        lot_value_estimate: Optional[float] = None,
        estimated_weight_kg: Optional[float] = None,
        price_per_kg: Optional[float] = None,
        currency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a structured recommendation for the current batch."""

        resolved_currency = self._normalize_currency(currency or self.currency)
        total_leaves = max(healthy_count + unhealthy_count, 0)
        completion_rate = round((completed_images / total_images) * 100, 1) if total_images > 0 else 100.0

        if total_leaves <= 0:
            return {
                "status": "insufficient_data",
                "session_id": session_id,
                "session_name": session_name,
                "summary": {
                    "total_images": total_images,
                    "completed_images": completed_images,
                    "completion_rate": completion_rate,
                    "healthy_leaves": healthy_count,
                    "unhealthy_leaves": unhealthy_count,
                    "total_leaves": total_leaves,
                    "health_percentage": 0.0,
                    "quality_grade": "n/a",
                    "risk_level": "unknown",
                },
                "batch_decision": {
                    "action": "process",
                    "action_label": "Scan more leaves",
                    "reason": "No usable leaves were detected yet, so a sell/process/discard recommendation is not reliable.",
                    "confidence": 0.0,
                    "allocation": {"sell": 0, "process": 0, "discard": 0},
                },
                "economic_impact": {
                    "currency": resolved_currency,
                    "estimate_mode": "insufficient_data",
                    "estimated_gross_value": 0.0,
                    "estimated_recoverable_value": 0.0,
                    "estimated_value_at_risk": 0.0,
                    "estimated_economic_loss_percentage": 0.0,
                    "health_loss_percentage": 0.0,
                },
                "value_recovery": {
                    "recovery_priority": "Capture more clear leaf images before deciding how to route the batch.",
                    "salvageable_share": 0.0,
                    "routes": [],
                },
                "actions": {
                    "immediate": [
                        "Rescan the batch with clear leaf coverage.",
                        "Make sure the camera is focused on the leaf surface.",
                        "Collect a larger sample before routing the lot.",
                    ],
                    "next_24h": [
                        "Retake the scan after sorting the lot into a visible layer.",
                        "Record field, picking time, and storage conditions for the next scan.",
                    ],
                    "monitoring": [
                        "Track scan quality and coverage.",
                        "Check whether leaf overlap is hiding defects.",
                    ],
                },
                "assumptions": {
                    "currency": resolved_currency,
                    "value_per_leaf": self.value_per_leaf,
                    "default_price_per_kg": self.default_price_per_kg,
                    "assumed_leaves_per_kg": self.assumed_leaves_per_kg,
                    "minimum_sample_weight_kg": self.minimum_sample_weight_kg,
                    "estimate_mode": "insufficient_data",
                    "calibration_note": "Add more scan data, or provide lot weight and price/kg, to estimate value more accurately.",
                },
            }

        health_percentage = round((healthy_count / total_leaves) * 100, 1)
        quality_grade = self._quality_grade(health_percentage)
        risk_level = self._risk_level(health_percentage)
        action, action_label = self._recommended_action(quality_grade)
        allocation = self._allocation_profile(quality_grade)
        recovery_rate = self._recovery_rate(quality_grade)
        base_confidence = self._base_confidence(quality_grade)
        decision_confidence = round(base_confidence * self._coverage_weight(completion_rate), 2)
        estimated_weight_kg = max(
            estimated_weight_kg if estimated_weight_kg is not None else total_leaves / self.assumed_leaves_per_kg,
            self.minimum_sample_weight_kg,
        )
        healthy_weight_kg = round(estimated_weight_kg * (health_percentage / 100), 3)
        unhealthy_weight_kg = round(max(estimated_weight_kg - healthy_weight_kg, 0.0), 3)

        if health_percentage >= 85:
            sale_grade = "premium"
            sale_price_per_kg = self.premium_price_per_kg
        elif health_percentage >= 60:
            sale_grade = "standard"
            sale_price_per_kg = self.standard_price_per_kg
        else:
            sale_grade = "reject"
            sale_price_per_kg = self.reject_price_per_kg

        ideal_revenue = round(estimated_weight_kg * self.premium_price_per_kg, 2)
        actual_revenue = round(estimated_weight_kg * sale_price_per_kg, 2)
        estimated_income_loss = round(max(ideal_revenue - actual_revenue, 0.0), 2)

        recoverable_weight_kg = round(unhealthy_weight_kg * recovery_rate, 3)
        waste_before_kg = round(unhealthy_weight_kg, 3)
        waste_after_kg = round(max(waste_before_kg - recoverable_weight_kg, 0.0), 3)
        waste_reduction_kg = round(recoverable_weight_kg, 3)
        waste_reduction_pct = round((waste_reduction_kg / waste_before_kg * 100), 1) if waste_before_kg > 0 else 0.0

        recovered_value = round(recoverable_weight_kg * max(self.standard_price_per_kg - self.reject_price_per_kg, 0.0), 2)
        sorting_cost = round(estimated_weight_kg * self.sorting_cost_per_kg, 2)
        rework_cost = round(recoverable_weight_kg * self.rework_cost_per_kg, 2)
        residual_disposal_cost = round(waste_after_kg * self.disposal_cost_per_kg, 2)
        baseline_disposal_cost = round(waste_before_kg * self.disposal_cost_per_kg, 2)
        potential_savings_from_waste_reduction = round(max(recovered_value - sorting_cost - rework_cost, 0.0), 2)

        profit_before_action = round(actual_revenue - baseline_disposal_cost, 2)
        profit_after_action = round(actual_revenue + recovered_value - sorting_cost - rework_cost - residual_disposal_cost, 2)
        profit_improvement_per_batch = round(profit_after_action - profit_before_action, 2)
        profit_improvement_percentage = round((profit_improvement_per_batch / profit_before_action) * 100, 1) if profit_before_action > 0 else 0.0

        estimated_gross_value, estimate_mode = self._estimate_gross_value(
            total_leaves=total_leaves,
            lot_value_estimate=lot_value_estimate,
            estimated_weight_kg=estimated_weight_kg,
            price_per_kg=price_per_kg,
        )
        estimated_recoverable_value = round(estimated_gross_value * recovery_rate, 2)
        estimated_value_at_risk = round(max(estimated_gross_value - estimated_recoverable_value, 0.0), 2)
        economic_loss_percentage = round((estimated_value_at_risk / estimated_gross_value) * 100, 1) if estimated_gross_value > 0 else 0.0

        return {
            "status": "partial" if completion_rate < 100.0 else "ready",
            "session_id": session_id,
            "session_name": session_name,
            "summary": {
                "total_images": total_images,
                "completed_images": completed_images,
                "completion_rate": completion_rate,
                "healthy_leaves": healthy_count,
                "unhealthy_leaves": unhealthy_count,
                "total_leaves": total_leaves,
                "health_percentage": health_percentage,
                "quality_grade": quality_grade,
                "risk_level": risk_level,
            },
            "batch_decision": {
                "action": action,
                "action_label": action_label,
                "reason": self._decision_reason(quality_grade, health_percentage, unhealthy_count),
                "confidence": decision_confidence,
                "allocation": allocation,
            },
            "economic_impact": {
                "currency": resolved_currency,
                "estimate_mode": estimate_mode,
                "estimated_weight_kg": estimated_weight_kg,
                "healthy_weight_kg": healthy_weight_kg,
                "unhealthy_weight_kg": unhealthy_weight_kg,
                "sale_grade": sale_grade,
                "sale_price_per_kg": sale_price_per_kg,
                "ideal_revenue": ideal_revenue,
                "actual_revenue": actual_revenue,
                "estimated_income_loss": estimated_income_loss,
                "waste_before_kg": waste_before_kg,
                "waste_after_kg": waste_after_kg,
                "waste_reduction_kg": waste_reduction_kg,
                "waste_reduction_pct": waste_reduction_pct,
                "recovered_value": recovered_value,
                "sorting_cost": sorting_cost,
                "rework_cost": rework_cost,
                "residual_disposal_cost": residual_disposal_cost,
                "baseline_disposal_cost": baseline_disposal_cost,
                "potential_savings_from_waste_reduction": potential_savings_from_waste_reduction,
                "profit_before_action": profit_before_action,
                "profit_after_action": profit_after_action,
                "profit_improvement_per_batch": profit_improvement_per_batch,
                "profit_improvement_pct": profit_improvement_percentage,
                "estimated_gross_value": estimated_gross_value,
                "estimated_recoverable_value": estimated_recoverable_value,
                "estimated_value_at_risk": estimated_value_at_risk,
                "estimated_economic_loss_percentage": economic_loss_percentage,
                "health_loss_percentage": round(100 - health_percentage, 1),
            },
            "value_recovery": {
                "recovery_priority": self._recovery_priority(quality_grade),
                "salvageable_share": round(recovery_rate * 100, 1),
                "routes": self._recovery_routes(quality_grade),
            },
            "actions": {
                "immediate": self._immediate_actions(quality_grade),
                "next_24h": self._next_24h_actions(quality_grade),
                "monitoring": self._monitoring_points(quality_grade),
            },
            "assumptions": {
                "currency": resolved_currency,
                "value_per_leaf": self.value_per_leaf,
                "default_price_per_kg": self.default_price_per_kg,
                "premium_price_per_kg": self.premium_price_per_kg,
                "standard_price_per_kg": self.standard_price_per_kg,
                "reject_price_per_kg": self.reject_price_per_kg,
                "sorting_cost_per_kg": self.sorting_cost_per_kg,
                "rework_cost_per_kg": self.rework_cost_per_kg,
                "disposal_cost_per_kg": self.disposal_cost_per_kg,
                "assumed_leaves_per_kg": self.assumed_leaves_per_kg,
                "minimum_sample_weight_kg": self.minimum_sample_weight_kg,
                "estimate_mode": estimate_mode,
                "calibration_note": "Provide batch weight and price/kg for more precise currency estimates.",
            },
        }

    def _safe_float(self, value: Optional[str], default: float) -> float:
        try:
            if value is None:
                return default
            parsed = float(value)
            return parsed if parsed >= 0 else default
        except (TypeError, ValueError):
            return default

    def _normalize_currency(self, currency: Optional[str]) -> str:
        if not currency:
            return "USD"
        normalized = currency.strip().upper()
        return normalized[:3] if len(normalized) >= 3 else "USD"

    def _quality_grade(self, health_percentage: float) -> str:
        if health_percentage >= 85:
            return "premium"
        if health_percentage >= 70:
            return "standard"
        if health_percentage >= 45:
            return "below_standard"
        return "reject"

    def _risk_level(self, health_percentage: float) -> str:
        if health_percentage >= 85:
            return "low"
        if health_percentage >= 70:
            return "moderate"
        if health_percentage >= 45:
            return "high"
        return "critical"

    def _recommended_action(self, quality_grade: str) -> Tuple[str, str]:
        if quality_grade == "premium":
            return "sell", "Sell immediately as premium tea"
        if quality_grade == "standard":
            return "sell", "Sell after a quick re-sort"
        if quality_grade == "below_standard":
            return "process", "Process into a lower-grade product"
        return "discard", "Discard or compost the batch"

    def _allocation_profile(self, quality_grade: str) -> Dict[str, int]:
        profiles = {
            "premium": {"sell": 90, "process": 8, "discard": 2},
            "standard": {"sell": 70, "process": 20, "discard": 10},
            "below_standard": {"sell": 15, "process": 55, "discard": 30},
            "reject": {"sell": 0, "process": 25, "discard": 75},
        }
        return profiles.get(quality_grade, {"sell": 0, "process": 0, "discard": 0})

    def _recovery_rate(self, quality_grade: str) -> float:
        return {
            "premium": 0.95,
            "standard": 0.82,
            "below_standard": 0.58,
            "reject": 0.30,
        }.get(quality_grade, 0.0)

    def _base_confidence(self, quality_grade: str) -> float:
        return {
            "premium": 0.93,
            "standard": 0.87,
            "below_standard": 0.80,
            "reject": 0.76,
        }.get(quality_grade, 0.6)

    def _coverage_weight(self, completion_rate: float) -> float:
        if completion_rate >= 100:
            return 1.0
        if completion_rate <= 0:
            return 0.5
        return round(0.7 + (completion_rate / 100) * 0.3, 2)

    def _decision_reason(self, quality_grade: str, health_percentage: float, unhealthy_count: int) -> str:
        if quality_grade == "premium":
            return (
                f"{health_percentage:.1f}% of detected leaves are healthy, so the batch should be sold quickly "
                "to capture the best market price. Keep the premium fraction separate from lower-grade material."
            )
        if quality_grade == "standard":
            return (
                f"The batch is still mostly saleable at {health_percentage:.1f}% health, but the damaged portion "
                f"({unhealthy_count} leaves) should be sorted out before sale to avoid price penalties."
            )
        if quality_grade == "below_standard":
            return (
                f"The batch has a noticeable defect rate ({unhealthy_count} unhealthy leaves), so it is more profitable "
                "to process the usable portion into a lower-grade product than to sell it as premium tea."
            )
        return (
            f"Only {health_percentage:.1f}% of the batch is healthy, so the lot should be diverted away from premium "
            "sales and routed to recovery or compost to avoid wasting processing time and buyer trust."
        )

    def _recovery_priority(self, quality_grade: str) -> str:
        return {
            "premium": "Protect the premium fraction and sell it first.",
            "standard": "Re-sort to preserve sale value before processing the leftovers.",
            "below_standard": "Split the batch into salvageable tea and low-value residue.",
            "reject": "Divert the batch to compost or biomass recovery to recover soil value.",
        }.get(quality_grade, "Review the batch before making a routing decision.")

    def _recovery_routes(self, quality_grade: str) -> List[Dict[str, str]]:
        routes = {
            "premium": [
                {
                    "title": "Keep premium leaves separate",
                    "description": "Seal and label the healthiest fraction so it can be sold immediately at the highest price.",
                    "priority": "high",
                },
                {
                    "title": "Re-grade the remaining leaves",
                    "description": "Move visibly damaged leaves into a lower-grade stream before the buyer sees the lot.",
                    "priority": "medium",
                },
                {
                    "title": "Use final rejects for compost",
                    "description": "Only the unusable residue should be sent to compost or biomass recovery.",
                    "priority": "low",
                },
            ],
            "standard": [
                {
                    "title": "Re-sort and sell quickly",
                    "description": "A quick hand sort can preserve the premium part of the batch and reduce price cuts.",
                    "priority": "high",
                },
                {
                    "title": "Route damaged but safe leaves to lower-grade products",
                    "description": "Use the usable remainder for tea bags, blended tea, or other lower-price channels.",
                    "priority": "medium",
                },
                {
                    "title": "Recover soil value from the final residue",
                    "description": "Compost or mulch the final residue so waste leaves still support farm fertility.",
                    "priority": "low",
                },
            ],
            "below_standard": [
                {
                    "title": "Extract the salvageable fraction",
                    "description": "Separate the cleanest leaves so they can still enter a lower-grade sale channel.",
                    "priority": "high",
                },
                {
                    "title": "Process into tea bags or blends",
                    "description": "Convert the middle-quality fraction into products where appearance matters less than consistency.",
                    "priority": "medium",
                },
                {
                    "title": "Compost or biomass the rest",
                    "description": "Turn the unusable remainder into compost, mulch, or another non-saleable recovery stream.",
                    "priority": "low",
                },
            ],
            "reject": [
                {
                    "title": "Avoid premium mixing",
                    "description": "Do not mix the lot with saleable tea because it will drag down the entire batch price.",
                    "priority": "high",
                },
                {
                    "title": "Recover only safe secondary value",
                    "description": "If the leaves are food-safe, send them to compost, mulch, or biomass recovery.",
                    "priority": "medium",
                },
                {
                    "title": "Investigate the root cause",
                    "description": "Check picking, storage, and field conditions before the next harvest to prevent repeat losses.",
                    "priority": "low",
                },
            ],
        }
        return routes.get(quality_grade, [])

    def _immediate_actions(self, quality_grade: str) -> List[str]:
        actions = {
            "premium": [
                "Keep this batch separate from lower-grade leaves.",
                "Move the saleable fraction to packing or buyer pickup first.",
                "Label the lot so premium value is not diluted.",
            ],
            "standard": [
                "Do a fast re-sort before sale.",
                "Keep damaged leaves out of the premium lot.",
                "Pack the cleanest fraction immediately to preserve freshness.",
            ],
            "below_standard": [
                "Separate the batch into salvageable and unusable leaves.",
                "Send the usable fraction to a lower-grade processing line.",
                "Prevent the lot from being sold as premium tea.",
            ],
            "reject": [
                "Do not mix this lot with saleable tea.",
                "Route safe residue to compost or biomass recovery.",
                "Review picking, drying, and storage steps before the next batch.",
            ],
        }
        return actions.get(quality_grade, [])

    def _next_24h_actions(self, quality_grade: str) -> List[str]:
        actions = {
            "premium": [
                "Finalise sale documentation and buyer communication.",
                "Track the sell-through rate for this batch.",
            ],
            "standard": [
                "Measure how much value was preserved after re-sorting.",
                "Record what caused the damaged fraction for future improvement.",
            ],
            "below_standard": [
                "Map which leaves were still salvageable and why.",
                "Decide the best lower-grade product channel before storage quality drops further.",
            ],
            "reject": [
                "Document the reject rate by field or picker.",
                "Check whether weather, delay, or handling caused the quality drop.",
            ],
        }
        return actions.get(quality_grade, [])

    def _monitoring_points(self, quality_grade: str) -> List[str]:
        return {
            "premium": [
                "Watch how quickly the premium fraction reaches market.",
                "Track whether a later sort reduces the premium share.",
            ],
            "standard": [
                "Track reject rate by field and picker.",
                "Compare value preserved before and after re-sorting.",
            ],
            "below_standard": [
                "Watch moisture and bruising during collection.",
                "Track how much of the batch can still be sold in lower-grade channels.",
            ],
            "reject": [
                "Investigate storage delay, contamination, and over-mature leaf pickup.",
                "Check whether training or equipment issues are driving the reject rate.",
            ],
        }.get(quality_grade, [])

    def _estimate_gross_value(
        self,
        *,
        total_leaves: int,
        lot_value_estimate: Optional[float],
        estimated_weight_kg: Optional[float],
        price_per_kg: Optional[float],
    ) -> Tuple[float, str]:
        if lot_value_estimate is not None:
            return round(max(lot_value_estimate, 0.0), 2), "lot_value_provided"

        if estimated_weight_kg is not None and price_per_kg is not None:
            gross_value = max(estimated_weight_kg, 0.0) * max(price_per_kg, 0.0)
            return round(gross_value, 2), "weight_and_price_provided"

        if estimated_weight_kg is not None:
            gross_value = max(estimated_weight_kg, 0.0) * max(self.default_price_per_kg, 0.0)
            return round(gross_value, 2), "weight_based"

        if price_per_kg is not None and self.assumed_leaves_per_kg > 0:
            assumed_weight = max(total_leaves / self.assumed_leaves_per_kg, self.minimum_sample_weight_kg)
            gross_value = assumed_weight * max(price_per_kg, 0.0)
            return round(gross_value, 2), "sample_weight_estimate"

        gross_value = max(total_leaves, 0) * self.value_per_leaf
        return round(gross_value, 2), "sample_based"