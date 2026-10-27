"""Plain-language layer: what every statistic means and how to act on it.

Kept apart from the maths so the wording can be edited without touching a
calculation. The interface reads from here for tooltips, the glossary, the
"what this means" line under each validation check, and the one-paragraph
summary of a validation run.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Hover help for table columns (st.dataframe column_config)
# ----------------------------------------------------------------------
COLUMN_HELP: Dict[str, str] = {
    "Verdict": "A plain-words reading of the row. See the rules in the Guide tab.",
    "Mean IC": "Average daily rank correlation between the factor and what the instruments returned next. "
               "0 = no predictive power, +1 = perfect. Anything steadily above +0.05 is notable.",
    "ICIR": "Mean IC divided by how much the IC bounces around. Higher = the signal works more consistently, "
            "not just on average.",
    "t-stat": "Mean IC measured in standard errors. About 2 is 'significant' for ONE test; "
              "when many factors are screened you need to look at the adjusted columns instead.",
    "p": "Chance of seeing a result this strong if the factor had no predictive power at all. "
         "Smaller is better, but it ignores how many factors you looked at.",
    "p_bonferroni": "The p-value multiplied by the number of factors screened. Very strict: "
                    "below 0.05 means the factor survives even a harsh correction.",
    "p_bh": "The p-value corrected for having screened many factors (Benjamini-Hochberg). "
            "Below 0.10 means it survives with roughly a 10% false-discovery rate. THIS is the column to trust.",
    "Spread Sharpe": "Sharpe ratio of buying the top-scored bucket and shorting the bottom one every period. "
                     "Shows whether the ranking turns into money. You cannot always short, so it is a test, "
                     "not a tradable result.",
    "Monotonicity": "Do the score buckets earn in order? +1 = each higher bucket earned more than the one below. "
                    "A factor that only works at the extremes scores low.",
    "Top-bucket turnover": "Share of the top bucket replaced at each rebalance. High turnover means high trading costs.",
    "n": "Number of independent (non-overlapping) observations behind the statistics.",
    "Weight": "Share of the portfolio's capital in this holding.",
    "Stand-alone vol": "How volatile this holding is on its own, per year.",
    "Risk contribution": "Share of the PORTFOLIO's volatility that comes from this holding, including how it "
                         "moves with the others. Shares add to 100%.",
    "Return contribution": "Share of the average annual return this holding supplied.",
    "Risk / weight": "Risk share divided by weight share. Above 1 = this holding punches above its weight in risk.",
    "CAR": "Cumulative abnormal return: how much more (or less) the security earned across the event window "
           "than the market model predicted.",
    "CAR t": "That abnormal return measured in standard errors. About 2 or more is notable for one event.",
    "Beta": "Sensitivity to the market proxy, estimated before the event.",
    "Day-0 AR": "Abnormal return on the event day itself.",
}

# ----------------------------------------------------------------------
# Glossary, grouped for the Guide tab: (term, plain meaning, what good looks like, watch out for)
# ----------------------------------------------------------------------
GLOSSARY: Dict[str, List[Tuple[str, str, str, str]]] = {
    "Factor Lab": [
        ("Factor", "A number computed for every instrument every day from its price history (for example its "
                   "return over the last 12 months) that you suspect predicts what it does next.",
         "A factor with a believable economic reason.", "A factor you only believe because it worked in this data."),
        ("Forward horizon", "How far ahead the factor is asked to predict (21 days is about a month).",
         "Close to how often you would really rebalance.", "Picking the horizon that flatters one factor."),
        ("IC (information coefficient)", "Each day: rank the instruments by the factor, rank them by what they "
                                         "actually returned afterwards, and correlate the two rankings. "
                                         "+1 = perfect foresight, 0 = no relationship, negative = backwards.",
         "Steady +0.03 to +0.10. Bigger is rare and deserves suspicion.", "Tiny universes make IC jumpy."),
        ("ICIR", "Mean IC divided by the spread of the daily ICs: a signal-to-noise ratio.",
         "Above about 0.3 on this scale is solid.", "A high mean IC with a low ICIR means it works in bursts."),
        ("t-stat", "The mean IC measured in standard errors (Newey-West adjusted for overlapping windows).",
         "Above 3 once you have screened many factors.", "|t| near 2 is what pure noise produces regularly "
                                                          "when you test 40 things."),
        ("p-value / p_bh / p_bonferroni", "Probability of a result this strong from pure luck. p_bh and "
                                          "p_bonferroni correct for the number of factors screened.",
         "p_bh below 0.10 (survives) or below 0.05 (strong).", "A low raw p with a high p_bh is luck dressed up."),
        ("Long-short spread Sharpe", "Sharpe of holding the best bucket long and the worst short each period.",
         "Positive and in line with the IC.", "Not directly tradable if you cannot short."),
        ("Monotonicity", "Do buckets earn in order, worst bucket lowest, best highest?",
         "Close to +1.", "Works only in one tail = fragile."),
        ("Top-bucket turnover", "How much of the best bucket changes at each rebalance.",
         "Below about 50% keeps costs modest.", "Fast signals can lose everything to trading costs."),
        ("IC by horizon (decay)", "The IC measured at 1, 5, 10, 21, 42 and 63 days ahead.",
         "Peaks near the horizon you will trade at.", "A signal that is gone in 5 days is no use to a monthly rebalance."),
        ("Multiple-testing correction", "Screening 43 factors at the 5% level hands you about two false winners "
                                        "by chance. The correction shrinks every p-value by how many you tried.",
         "Use the 'Survive FDR' count, not the raw winners.", "Choosing the best-looking raw winner after the fact."),
    ],
    "Strategy validation": [
        ("Sharpe ratio", "Average return per unit of volatility, annualised.", "Above about 0.7 after costs is decent.",
         "It is an estimate with wide error bars (see the bootstrap)."),
        ("Probabilistic Sharpe (PSR)", "The probability the TRUE Sharpe is above zero, allowing for the sample "
                                       "length and for skewed, fat-tailed returns.",
         "95% or more.", "A short record with a high Sharpe can still be luck."),
        ("Deflated Sharpe (DSR)", "The PSR after charging for every configuration you tried. The more variants "
                                  "you test, the higher the Sharpe luck alone would have produced.",
         "95% or more even after your trial count.", "Shows N/A when no hypothesis is selected, because the "
                                                     "app cannot know how many things you tried."),
        ("Minimum track record", "How many years of data are needed before this Sharpe is distinguishable "
                                 "from zero at 95% confidence.", "Less than the history you have.",
         "N/A means the Sharpe is not even positive."),
        ("Bootstrap interval", "Re-draws your return history in blocks thousands of times to see how much the "
                               "Sharpe would vary. The 95% interval is the plausible range.",
         "The whole interval above zero.", "A wide interval means you do not really know the Sharpe."),
        ("Sign-flip test", "Randomly flips the sign of each day's return. If your result still beats most "
                           "of those scrambles, the average return is real rather than noise.",
         "p below 0.05.", "It says there is a positive mean, not that the rule caused it."),
        ("Random-timing test", "Keeps your exact positions but rotates them in time against the returns. If "
                               "that scores as well, the profit came from what you owned, not when you owned it.",
         "p below 0.05 for a rule that claims timing skill.", "Fails for always-invested strategies on a rising "
                                                             "market: that return is mostly market exposure (beta)."),
        ("Walk-forward consistency", "Splits the history into five equal periods and checks the Sharpe in each.",
         "At least four of five positive.", "One great period carrying four mediocre ones."),
        ("Second half vs first half", "Does the second half of the history earn as much as the first?",
         "60%+ of the first-half Sharpe retained.", "A strategy that worked early and faded."),
        ("Cost stress", "Reruns the strategy with commissions and slippage doubled.", "Keeps 70%+ of the Sharpe.",
         "High-turnover strategies often collapse here."),
        ("PBO (probability of backtest overfitting)", "Across a grid of parameter settings, how often the best "
                                                      "setting in one half of history ranks below average in the other half.",
         "Below 20%.", "Around 50% means picking the best backtest is no better than picking at random."),
        ("CPCV (combinatorial purged cross-validation)", "Repeatedly picks the best parameters on part of the "
                                                         "history and scores them on held-out parts, with gaps "
                                                         "so nothing leaks across.",
         "Median out-of-sample Sharpe positive and few losing splits.", "A big gap between in-sample and "
                                                                      "out-of-sample Sharpe."),
        ("Trial count", "How many configurations you tried under one hypothesis, from the ledger.",
         "Honest and complete.", "Forgetting the dead ends. That is the main way backtests mislead."),
        ("Grade", "Robust / Promising / Fragile / Unsupported: a weighted tally of how many independent checks "
                  "agree. A failed deflated Sharpe, PBO, PSR or bootstrap caps it.",
         "Promising or better, then monitor it.", "It is not a probability of making money."),
    ],
    "Event study": [
        ("Event", "A dated occurrence: an earnings release, a rate decision, a market crash, a rebalance.", "", ""),
        ("Event window", "The trading days before and after the event that are examined (day 0 = the event).",
         "Long enough to see the reaction, short enough to stay about the event.", ""),
        ("Estimation window", "About 200 days ending shortly before the event window, used to learn how the "
                              "security normally moves with the market.", "Free of other big events.", ""),
        ("Abnormal return (AR)", "What the security earned on a day minus what the market model said it "
                                 "should have earned.", "", ""),
        ("CAR", "The abnormal returns added up across the window.", "Large relative to its noise.", ""),
        ("t-stat of CAR", "The average CAR over all events, in standard errors.", "Above about 2 with 10+ events.",
         "With only a handful of events treat the average as an anecdote."),
        ("Share positive", "Fraction of events with a positive CAR.", "Well away from 50%.", ""),
    ],
    "Risk x-ray": [
        ("Risk contribution", "How much of the portfolio's volatility each holding is responsible for, "
                              "counting how it moves with the rest.", "Roughly matches your intent.",
         "One holding with 15% of the money and 50% of the risk."),
        ("Effective N (weights)", "How many equal holdings the weights are equivalent to (1 / HHI).",
         "Close to the number you hold.", "Much lower means the weights are concentrated."),
        ("Effective bets (risk)", "How many truly independent sources of risk you have "
                                  "(diversification ratio squared).", "Close to the number of holdings.",
         "Three funds that all move together are one bet."),
        ("Diversification ratio", "Weighted average stand-alone volatility divided by portfolio volatility.",
         "Well above 1.", "Near 1 means no diversification benefit."),
        ("VaR / expected shortfall", "Loss that is exceeded on the worst 5% (or 1%) of days, and the "
                                     "average loss on those days.", "", "Computed from the sample, so "
                                                                        "it understates what has not happened yet."),
        ("Cornish-Fisher VaR", "VaR adjusted for the skew and fat tails seen in the sample.", "", ""),
    ],
    "Monitor": [
        ("Registered strategy", "A strategy frozen on a date. Data after that date is genuinely out-of-sample.",
         "", ""),
        ("z vs baseline", "How many standard errors the forward average return is from what the "
                          "registration-time Sharpe implied.", "Above -1.", "Below -2 is flagged as stale."),
        ("CUSUM", "A running tally that accumulates evidence of persistent under-performance and resets "
                  "when results recover.", "Staying near zero.", "A breach is a prompt to investigate."),
        ("fresh / healthy / aging / stale", "Too early to say / consistent with the baseline / trailing but "
                                            "within chance / improbably weak.", "", "A prompt to look, never an "
                                                                                    "instruction to stop."),
    ],
}

# ----------------------------------------------------------------------
# Per-check wording: (what it asks, advice if it fails or warns)
# ----------------------------------------------------------------------
CHECK_HELP: Dict[str, Tuple[str, str]] = {
    "length": ("Is there enough history to say anything?",
               "Extend the start date or treat the result as preliminary."),
    "psr": ("Allowing for skew and sample size, how sure are we the true Sharpe is above zero?",
            "The edge is not statistically clear: a longer history or a stronger signal is needed."),
    "dsr": ("After the number of configurations tried, is the Sharpe still better than luck?",
            "Too many variants for this Sharpe: cut the search space, or demand the result hold out-of-sample."),
    "bootstrap": ("How wide is the plausible range for the Sharpe?",
                  "The Sharpe could easily be zero or negative. Do not size this as if the headline number were certain."),
    "sign_flip": ("Is the average return distinguishable from random noise?",
                  "The positive average return could be luck."),
    "timing": ("Does it add value by WHEN it holds things, or only by WHAT it holds?",
               "Most of the return is market exposure. Judge it against a simple buy-and-hold of the same assets."),
    "walk_forward": ("Does it work in most periods or only in one or two?",
                     "The edge is concentrated in a few periods. Check what happened there."),
    "oos": ("Does the second half of history hold up against the first?",
            "It may have decayed. Look at whether the reason it worked still applies."),
    "costs": ("Does it survive doubled trading costs?",
              "Trading costs eat the edge. Lower the turnover (longer rebalance interval, buffers) or drop it."),
    "pbo": ("If you pick the best parameters in one half, do they stay good in the other?",
            "Choosing the best parameters here is no better than guessing. Prefer a wide, flat zone of settings."),
    "cpcv": ("Out-of-sample, how does 'pick the best parameters' actually perform?",
             "Parameter selection is not delivering out-of-sample. Simplify the rule or fix the parameters in advance."),
}


def check_question(key: str) -> str:
    return CHECK_HELP.get(key, ("", ""))[0]


def plain_summary(validation: Dict[str, Any]) -> str:
    """One paragraph: how many checks pass, what is wrong, what to do."""
    checks = validation["checks"]
    g = validation["grade"]
    scored = [c for c in checks if c.status in ("pass", "warn", "fail")]
    passed = [c for c in scored if c.status == "pass"]
    bad = [c for c in scored if c.status in ("fail", "warn")]
    head = f"{len(passed)} of {len(scored)} checks pass, so the grade is {g['label']}. "
    if not bad:
        return head + "Nothing in the battery argues against it. Register it and watch the forward record."
    fails = [c for c in bad if c.status == "fail"]
    parts = [head]
    if fails:
        parts.append("Failed: " + "; ".join(f"{c.title.lower()} ({CHECK_HELP.get(c.key, ('', ''))[1]})" for c in fails) + ". ")
    warns = [c for c in bad if c.status == "warn"]
    if warns:
        parts.append("Borderline: " + ", ".join(c.title.lower() for c in warns) + ". ")
    na = [c for c in checks if c.status == "na"]
    if any(c.key == "dsr" for c in na):
        parts.append("The deflated Sharpe was not computed: select a hypothesis (or enter the number of trials) so "
                     "it can charge you for the variants you tried. ")
    return "".join(parts).strip()


def factor_verdict(row: Dict[str, Any]) -> str:
    """Plain-words label for one row of the factor screen."""
    p, pbh, ic = row.get("p"), row.get("p_bh"), row.get("Mean IC")
    if p is None or pbh is None or ic is None or p != p or pbh != pbh or ic != ic:
        return "Not enough data"
    if pbh < 0.05:
        return "Strong evidence" if ic > 0 else "Works in reverse (strong)"
    if pbh < 0.10:
        return "Probable signal" if ic > 0 else "Works in reverse (probable)"
    if p < 0.05:
        return "Likely luck (fails correction)"
    return "No evidence"
