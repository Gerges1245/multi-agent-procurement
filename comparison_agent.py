import json
import os
from decimal import Decimal
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from test_scraping import ProductExtraction


def condition_check(actual, required):
    if not actual:
        return None

    value = actual.strip().lower()
    required = required.strip().lower()

    aliases = {
        "new": "new",
        "brand new": "new",
        "brand-new": "new",
        "جديد": "new",
        "جديدة": "new",
        "used": "used",
        "pre-owned": "used",
        "مستعمل": "used",
        "refurbished": "refurbished",
        "renewed": "refurbished",
        "مجدد": "refurbished",
    }

    normalized = aliases.get(value)
    if normalized is None:
        return None

    return normalized == aliases.get(required, required)


def evaluate_product(record, context):
    row = {
        "source_url": record["source_url"],
        "name": record.get("candidate_title", ""),
        "rank": None,
        "value_score": None,
    }

    if record.get("status") != "extracted":
        row.update({
            "status": "excluded",
            "reason": record.get(
                "error", "No usable product extraction available."
            ),
        })
        return row

    try:
        product = ProductExtraction.model_validate(record["extraction"])
    except ValueError as exc:
        row.update({
            "status": "excluded",
            "reason": f"Invalid extraction: {exc}",
        })
        return row

    if product.page_type != "product":
        row.update({
            "status": "excluded",
            "reason": "The extracted page is not a product page.",
        })
        return row

    request = context["purchase_request"]
    required_currency = context["company"]["currency"].strip().upper()
    currency = (
        product.currency.strip().upper()
        if product.currency else None
    )

    price = (
        Decimal(str(product.current_cash_price))
        if product.current_cash_price is not None else None
    )

    if price is not None and (not price.is_finite() or price <= 0):
        price = None

    budget = Decimal(str(request["max_unit_price"]))
    quantity = request["quantity"]

    currency_ok = (
        currency == required_currency if currency is not None else None
    )

    budget_ok = (
        price <= budget
        if price is not None and currency_ok is True else None
    )

    ram_ok = (
        product.ram_gb >= request["minimum_ram_gb"]
        if product.ram_gb is not None else None
    )

    ssd_ok = (
        product.ssd_gb >= request["minimum_ssd_gb"]
        if product.ssd_gb is not None else None
    )

    quantity_ok = (
        product.quantity_available >= quantity
        if product.quantity_available is not None else None
    )

    availability = (product.availability_text or "").lower()
    if any(
        phrase in availability
        for phrase in (
            "out of stock",
            "sold out",
            "غير متوفر",
            "غير متاح",
            "نفد المخزون",
        )
    ):
        quantity_ok = False

    checks = {
        "currency": currency_ok,
        "budget": budget_ok,
        "ram": ram_ok,
        "ssd": ssd_ok,
        "condition": condition_check(
            product.condition, request["condition"]
        ),
        "quantity": quantity_ok,
    }

    failed_checks = [
        name for name, passed in checks.items() if passed is False
    ]
    unknown_checks = [
        name for name, passed in checks.items() if passed is None
    ]

    if failed_checks:
        status = "ineligible"
    elif unknown_checks:
        status = "needs_confirmation"
    else:
        status = "eligible"

    total = (
        price * quantity
        if price is not None and currency_ok is True else None
    )

    row.update({
        "name": product.product_name or row["name"],
        "status": status,
        "checks": checks,
        "failed_checks": failed_checks,
        "unknown_checks": unknown_checks,
        "unit_price": (
            str(price.quantize(Decimal("0.01")))
            if price is not None else None
        ),
        "currency": currency,
        "listed_total_for_requested_quantity": (
            str(total.quantize(Decimal("0.01")))
            if total is not None else None
        ),
        "ram_gb": product.ram_gb,
        "ssd_gb": product.ssd_gb,
        "processor": product.processor,
        "condition": product.condition,
        "quantity_available": product.quantity_available,
        "availability_text": product.availability_text,
        "warranty_text": product.warranty_text,
        "retrieved_at": record.get("retrieved_at"),
        "evidence_quotes": product.evidence_quotes,
        "limitations": product.limitations,
    })

    return row


def build_comparison(context, records):
    request = context["purchase_request"]

    if min(
        request["max_unit_price"],
        request["quantity"],
        request["minimum_ram_gb"],
        request["minimum_ssd_gb"],
    ) <= 0:
        raise ValueError("Budget, quantity, and minimum specs must be positive.")

    rows = [evaluate_product(record, context) for record in records]

    rankable = [
        row for row in rows
        if row["status"] in ("eligible", "needs_confirmation")
        and all(
            row["checks"][name] is True
            for name in ("currency", "budget", "ram", "ssd")
        )
    ]

    if rankable:
        lowest_price = min(
            Decimal(row["unit_price"]) for row in rankable
        )

        for row in rankable:
            price_score = 60 * float(
                lowest_price / Decimal(row["unit_price"])
            )

            ram_score = 20 * min(
                row["ram_gb"] / (2 * request["minimum_ram_gb"]),
                1,
            )

            ssd_score = 20 * min(
                row["ssd_gb"] / (2 * request["minimum_ssd_gb"]),
                1,
            )

            row["value_score"] = round(
                price_score + ram_score + ssd_score, 2
            )

        rankable.sort(
            key=lambda row: (
                row["status"] != "eligible",
                -row["value_score"],
                Decimal(row["unit_price"]),
            )
        )

        for rank, row in enumerate(rankable, start=1):
            row["rank"] = rank

    eligible = [
        row for row in rankable if row["status"] == "eligible"
    ]

    summary = {
        "total": len(rows),
        "eligible": sum(r["status"] == "eligible" for r in rows),
        "needs_confirmation": sum(
            r["status"] == "needs_confirmation" for r in rows
        ),
        "ineligible": sum(r["status"] == "ineligible" for r in rows),
        "excluded": sum(r["status"] == "excluded" for r in rows),
        "ranked": len(rankable),
        "best_eligible_url": (
            eligible[0]["source_url"] if eligible else None
        ),
    }

    return {
        "company_context": context,
        "summary": summary,
        "ranking_policy": {
            "weights": {"price": 60, "ram": 20, "ssd": 20},
            "price_rule": "60 × lowest rankable price / product price",
            "spec_rule": (
                "Each spec earns up to 20 points at twice its minimum; "
                "higher values earn no additional points."
            ),
            "eligibility_rule": (
                "Confirmed eligible products rank before conditional ones. "
                "Products with failed requirements are not ranked."
            ),
            "cost_note": (
                "Listed totals equal unit price × requested quantity. "
                "Unconfirmed shipping, taxes, and bulk discounts are excluded."
            ),
            "scope_note": (
                "Scores compare price, RAM, and SSD only. "
                "They are not CPU benchmarks or purchase guarantees."
            ),
        },
        "products": rows,
    }


class EmptyInput(BaseModel):
    pass


class CompareProductsTool(BaseTool):
    name: str = "compare_extracted_products"
    description: str = (
        "Compare saved product extractions against company requirements, "
        "calculate listed totals, rank suitable candidates, and save results. "
        "Takes no arguments."
    )
    args_schema: type[BaseModel] = EmptyInput

    project_dir: Path = Field(exclude=True)
    comparison: dict = Field(default_factory=dict, exclude=True)

    def _run(self) -> str:
        if self.comparison:
            return json.dumps(self.comparison["summary"])

        context = json.loads(
            (self.project_dir / "company_context.json").read_text(
                encoding="utf-8-sig"
            )
        )

        extraction_file = (
            self.project_dir / "outputs" / "product_extractions.json"
        )
        extraction_data = json.loads(
            extraction_file.read_text(encoding="utf-8-sig")
        )

        records = extraction_data["products"]
        if not records:
            raise RuntimeError("No extraction records available.")

        comparison = build_comparison(context, records)

        output_path = (
            self.project_dir / "outputs" / "comparison_results.json"
        )
        output_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.comparison = comparison
        return json.dumps(comparison["summary"])


def main():
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env", override=True)

    model = os.environ["GEMINI_MODEL"].strip().removeprefix("models/")
    if not model.startswith("gemini/"):
        model = f"gemini/{model}"

    llm = LLM(
        model=model,
        api_key=os.environ["GEMINI_API_KEY"],
        timeout=90,
    )

    comparison_tool = CompareProductsTool(
        project_dir=root,
        result_as_answer=True,
    )

    analyst = Agent(
        role="Procurement Comparison Analyst",
        goal="Compare extracted products using explicit procurement rules.",
        backstory=(
            "You use the comparison tool for calculations and eligibility. "
            "You preserve uncertainty and never recommend a rejected product."
        ),
        llm=llm,
        tools=[comparison_tool],
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1,
        verbose=True,
    )

    task = Task(
        description=(
            "Call compare_extracted_products once with no arguments. "
            "Return its summary without changing calculations or statuses."
        ),
        expected_output="The comparison tool's factual summary.",
        agent=analyst,
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        memory=False,
        cache=False,
        verbose=True,
    )

    crew.kickoff()

    if not comparison_tool.comparison:
        raise RuntimeError("The comparison tool was not executed.")

    comparison = comparison_tool.comparison

    print("\nComparison Agent completed")
    print(json.dumps(comparison["summary"], indent=2))

    for row in comparison["products"]:
        print(f"\n{row['name']}")
        print(f"Status: {row['status']}")

        if row["status"] == "excluded":
            print(f"Reason: {row['reason']}")
            continue

        print(f"Price: {row['unit_price']} {row['currency']}")
        print(f"Failed checks: {row['failed_checks']}")
        print(f"Unknown checks: {row['unknown_checks']}")
        print(f"Rank: {row['rank']} | Score: {row['value_score']}")

    print("\nSaved to: outputs/comparison_results.json")


if __name__ == "__main__":
    main()