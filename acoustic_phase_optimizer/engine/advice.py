"""Plain-language findings derived from the metrics, shared by the UI and the report."""

from __future__ import annotations


def advise(metrics: dict | None, room: dict | None = None) -> list[dict]:
    """Return [{level: good|warning|serious|critical, title, text}], most important first."""
    out: list[dict] = []
    if not metrics:
        return [{"level": "warning", "title": "No active speakers",
                 "text": "Unmute or add a speaker to see coverage and alignment results."}]
    m = metrics

    echo = m.get("echo_pct") or 0.0
    if echo >= 25:
        out.append({"level": "critical", "title": "Speakers are not time-aligned",
                    "text": f"{echo:.0f}% of seats hear the same sound from two speakers more than 30 ms apart, "
                            "which is heard as an echo or a slap. Run Quick Align."})
    elif echo >= 5:
        out.append({"level": "serious", "title": "Late arrivals at some seats",
                    "text": f"{echo:.0f}% of seats get audible arrivals more than 30 ms apart, usually the back of a "
                            "delay speaker firing toward the stage. Lower that speaker or aim it away."})

    clarity = m.get("clarity")
    if clarity is not None:
        if clarity < 0:
            out.append({"level": "serious", "title": "Low clarity",
                        "text": f"On average the reverberant sound is {abs(clarity):.1f} dB louder than the direct sound. "
                                "Speech will be hard to follow. Use narrower speakers aimed at the seats, move speakers "
                                "closer to listeners, or add absorption."})
        elif clarity < 4:
            out.append({"level": "warning", "title": "Moderate clarity",
                        "text": f"Direct sound leads the reverberation by only {clarity:.1f} dB on average. "
                                "Fine for music; speech benefits from 6 dB or more."})

    cov = m.get("coverage") or 0.0
    if cov >= 4:
        out.append({"level": "serious", "title": "Uneven coverage",
                    "text": f"Direct level varies by about ±{cov:.1f} dB from seat to seat. Re-aim the mains or add fills "
                            "where the map is dark."})
    elif cov >= 2.5:
        out.append({"level": "warning", "title": "Coverage could be more even",
                    "text": f"Direct level varies by about ±{cov:.1f} dB from seat to seat; ±2 dB or better is a good target."})

    cancel = m.get("cancel_pct") or 0.0
    if cancel >= 15:
        out.append({"level": "serious", "title": "Speakers are cancelling each other",
                    "text": f"In {cancel:.0f}% of seat-and-frequency combinations the speakers lose more than 3 dB against "
                            "each other. Re-run the alignment and reduce the overlap between neighbouring speakers."})
    elif cancel >= 8:
        out.append({"level": "warning", "title": "Some cancellation between speakers",
                    "text": f"{cancel:.0f}% of seat-and-frequency combinations lose more than 3 dB to interference. "
                            "Some of this is unavoidable wherever two speakers cover the same seats."})

    tonal = m.get("tonal") or 0.0
    if tonal >= 2.0:
        out.append({"level": "warning", "title": "Tonal balance is off target",
                    "text": f"The audience-average response deviates {tonal:.1f} dB RMS from the target curve. Run Auto EQ."})

    if room:
        rt = room.get("rt60_mid") or 0.0
        if rt >= 2.0:
            out.append({"level": "warning", "title": "Very reverberant room",
                        "text": f"Predicted mid-band RT60 is {rt:.1f} s. Keep sound off the walls and ceiling, and expect "
                                "the room to be drier with an audience in it."})

    if not out:
        out.append({"level": "good", "title": "System is well aligned",
                    "text": "Coverage, timing and summation are all within good-practice targets."})
    return out
