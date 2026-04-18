def get_insurance_recommendation(severity, detected_parts, confidence):
    part_costs = {
        "front-bumper-dent": (8000, 25000),
        "front-bumper-scratch": (3000, 10000),
        "rear-bumper-dent": (8000, 25000),
        "bonnet-dent": (10000, 35000),
        "doorouter-dent": (8000, 30000),
        "doorouter-scratch": (3000, 12000),
        "fender-dent": (7000, 22000),
        "Headlight-Damage": (5000, 20000),
        "Taillight-Damage": (4000, 15000),
        "Front-Windscreen-Damage": (12000, 45000),
        "Rear-windscreen-Damage": (10000, 40000),
        "Sidemirror-Damage": (3000, 12000),
        "roof-dent": (15000, 50000),
        "pillar-dent": (20000, 80000),
        "quaterpanel-dent": (10000, 35000),
        "RunningBoard-Dent": (5000, 18000),
        "Signlight-Damage": (2000, 8000),
        "medium-Bodypanel-Dent": (8000, 25000),
        "paint-chip": (2000, 6000),
        "paint-trace": (1500, 5000),
    }

    severity_multiplier = {"minor": 1.0, "moderate": 1.6, "severe": 2.5}

    min_cost, max_cost = 0, 0
    breakdown = []

    for part in detected_parts:
        matched = False
        for key in part_costs:
            if key.lower() in part.lower() or part.lower() in key.lower():
                low, high = part_costs[key]
                m = severity_multiplier[severity]
                pmin, pmax = int(low * m), int(high * m)
                min_cost += pmin
                max_cost += pmax
                breakdown.append({"part": part, "min": pmin, "max": pmax})
                matched = True
                break
        if not matched:
            min_cost += 5000
            max_cost += 20000
            breakdown.append({"part": part, "min": 5000, "max": 20000})

    THRESHOLD = 25000
    should_claim = max_cost >= THRESHOLD or severity == "severe"

    if severity == "severe":
        recommendation = "🚨 CLAIM INSURANCE IMMEDIATELY"
        reason = "Severe structural damage detected. Repair costs will be high."
    elif severity == "moderate" and max_cost >= THRESHOLD:
        recommendation = "⚠️ RECOMMEND CLAIMING INSURANCE"
        reason = f"Estimated repair cost (₹{min_cost:,}–₹{max_cost:,}) exceeds deductible threshold."
    elif severity == "moderate":
        recommendation = "💡 CONSIDER SELF-REPAIR"
        reason = (
            f"Moderate damage but cost (₹{min_cost:,}–₹{max_cost:,}) may be manageable."
        )
    else:
        recommendation = "✅ SELF-REPAIR RECOMMENDED"
        reason = f"Minor damage. Estimated cost ₹{min_cost:,}–₹{max_cost:,}. Not worth filing a claim."

    return {
        "min_cost": min_cost,
        "max_cost": max_cost,
        "recommendation": recommendation,
        "reason": reason,
        "should_claim": should_claim,
        "breakdown": breakdown,
    }
