#!/usr/bin/env python3
"""Build an HTML contact sheet for LiDAR BEV alignment batch outputs.

The script reads a batch ``summary.json`` and selects samples from scenarios
with the largest residuals or explicitly requested scenario prefixes. If the
corresponding ``frame_XXXX/history_YY.png`` files have been downloaded, they are
embedded in the HTML. If not, the report still lists the exact expected image
paths and metrics so the missing samples can be fetched from the remote output.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_FIELDS = [
    "iou_gain",
    "iou_gain_raw",
    "before_iou",
    "after_iou",
    "before_iou_raw",
    "after_iou_raw",
    "before_iou_dynamic",
    "after_iou_dynamic",
    "before_iou_support",
    "after_iou_support",
    "refinement_dx",
    "refinement_dy",
    "refinement_translation_norm",
    "after_score_mode",
    "support_pixels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Path to batch summary.json.")
    parser.add_argument(
        "--batch-output-dir",
        type=Path,
        default=None,
        help="Batch output directory containing route subdirectories. Defaults to summary parent.",
    )
    parser.add_argument("--out-html", type=Path, required=True, help="HTML report path to write.")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario prefix to include, e.g. Town12_Rep0_1498. Can be repeated.",
    )
    parser.add_argument("--top-scenarios", type=int, default=5, help="Top scenarios by mean residual norm.")
    parser.add_argument("--per-scenario", type=int, default=8, help="Max samples shown per scenario.")
    parser.add_argument(
        "--sort-by",
        choices=["norm", "gain", "raw_gain", "abs_raw_gain"],
        default="norm",
        help="How to rank samples within each scenario.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="Metric fields to include in the card.",
    )
    return parser.parse_args()


def route_output_dir(batch_output_dir: Path, route_dir: str) -> Path:
    safe_route = "__".join(Path(route_dir).parts[-4:])
    return batch_output_dir / safe_route


def scenario_name(route_name: str) -> str:
    parts = route_name.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else route_name


def residual_norm(row: dict[str, object]) -> float:
    if "refinement_translation_norm" in row:
        return float(row["refinement_translation_norm"])
    dx = float(row.get("refinement_dx", 0.0))
    dy = float(row.get("refinement_dy", 0.0))
    return math.hypot(dx, dy)


def load_rows(summary_path: Path, batch_output_dir: Path) -> list[dict[str, object]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for route in summary.get("routes", []):
        route_dir = str(route.get("route_dir", ""))
        route_name = Path(route_dir).name
        route_out = route_output_dir(batch_output_dir, route_dir)
        scenario = scenario_name(route_name)
        for frame in route.get("frames", []):
            for result in frame.get("results", []):
                current_frame = int(result["current_frame"])
                history_offset = int(result["history_offset"])
                image_path = route_out / f"frame_{current_frame:04d}" / f"history_{history_offset:02d}.png"
                rows.append(
                    {
                        **result,
                        "route_dir": route_dir,
                        "route_name": route_name,
                        "scenario": scenario,
                        "image_path": image_path,
                        "image_exists": image_path.exists(),
                        "norm": residual_norm(result),
                    }
                )
    return rows


def select_scenarios(rows: list[dict[str, object]], explicit: list[str], top_scenarios: int) -> list[str]:
    if explicit:
        return explicit

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario"])].append(row)

    ranked = []
    for scenario, scenario_rows in grouped.items():
        mean_norm = sum(float(row["norm"]) for row in scenario_rows) / len(scenario_rows)
        ranked.append((mean_norm, scenario))
    return [scenario for _, scenario in sorted(ranked, reverse=True)[:top_scenarios]]


def sort_key(row: dict[str, object], sort_by: str) -> float:
    if sort_by == "gain":
        return -float(row.get("iou_gain", 0.0))
    if sort_by == "raw_gain":
        return -float(row.get("iou_gain_raw", 0.0))
    if sort_by == "abs_raw_gain":
        return abs(float(row.get("iou_gain_raw", 0.0)))
    return float(row["norm"])


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def metric_table(row: dict[str, object], fields: list[str]) -> str:
    lines = []
    for field in fields:
        if field not in row:
            continue
        lines.append(
            "<tr>"
            f"<th>{html.escape(field)}</th>"
            f"<td>{html.escape(format_value(row[field]))}</td>"
            "</tr>"
        )
    return "<table>" + "\n".join(lines) + "</table>"


def image_html(row: dict[str, object], out_html: Path) -> str:
    image_path = Path(row["image_path"])
    if bool(row["image_exists"]):
        try:
            rel = image_path.relative_to(out_html.parent)
        except ValueError:
            rel = image_path
        return f'<img src="{html.escape(str(rel))}" loading="lazy" />'
    return (
        '<div class="missing">'
        "image not downloaded<br/>"
        f"<code>{html.escape(str(image_path))}</code>"
        "</div>"
    )


def render_html(
    rows: list[dict[str, object]],
    selected_scenarios: list[str],
    out_html: Path,
    fields: list[str],
    per_scenario: int,
    sort_by: str,
) -> str:
    cards = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row["scenario"]) in selected_scenarios:
            grouped[str(row["scenario"])].append(row)

    for scenario in selected_scenarios:
        scenario_rows = sorted(grouped.get(scenario, []), key=lambda row: sort_key(row, sort_by), reverse=True)
        shown = scenario_rows[:per_scenario]
        mean_norm = sum(float(row["norm"]) for row in scenario_rows) / len(scenario_rows) if scenario_rows else 0.0
        cards.append(
            f'<section><h2>{html.escape(scenario)}</h2>'
            f'<p>samples={len(scenario_rows)} shown={len(shown)} mean_norm={mean_norm:.6g}</p>'
            '<div class="grid">'
        )
        for row in shown:
            title = (
                f"{row['route_name']} frame={int(row['current_frame']):04d} "
                f"history={int(row['history_offset']):02d}"
            )
            cards.append(
                '<article class="card">'
                f"<h3>{html.escape(title)}</h3>"
                f"{image_html(row, out_html)}"
                f"{metric_table(row, fields)}"
                "</article>"
            )
        cards.append("</div></section>")

    missing_count = sum(not bool(row["image_exists"]) for row in rows if str(row["scenario"]) in selected_scenarios)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LiDAR BEV Contact Sheet</title>
  <style>
    body {{
      margin: 24px;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f3ea;
      color: #191714;
    }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    section {{ margin-top: 32px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: #fffdf8;
      border: 1px solid #d8cfbf;
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 8px 24px rgb(60 45 20 / 10%);
    }}
    img {{
      display: block;
      width: 100%;
      image-rendering: pixelated;
      background: #111;
      border-radius: 8px;
      margin: 8px 0 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 4px 6px;
      border-bottom: 1px solid #eee4d2;
    }}
    th {{ width: 48%; color: #5b5142; }}
    code {{ word-break: break-all; }}
    .missing {{
      min-height: 96px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      background: #181612;
      color: #f4e8d2;
      border-radius: 8px;
      padding: 12px;
      margin: 8px 0 12px;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <h1>LiDAR BEV Contact Sheet</h1>
  <p>selected_scenarios={html.escape(', '.join(selected_scenarios))}</p>
  <p>missing_images={missing_count}. If images are missing, download the listed PNG paths from the remote output and rerun this script.</p>
  {''.join(cards)}
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    batch_output_dir = args.batch_output_dir or args.summary.parent
    rows = load_rows(args.summary, batch_output_dir)
    selected_scenarios = select_scenarios(rows, args.scenario, args.top_scenarios)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(
        render_html(rows, selected_scenarios, args.out_html, args.fields, args.per_scenario, args.sort_by),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_html}")
    print(f"Selected scenarios: {', '.join(selected_scenarios)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
