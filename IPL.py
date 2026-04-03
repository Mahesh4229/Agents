from pathlib import Path

import pandas as pd


TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
}

RECENT_MATCHES = 5
HEAD_TO_HEAD_MATCHES = 5


def find_csv_path() -> Path:
    csv_candidates = [
        Path(__file__).with_name("IPL.csv"),
        Path.home() / "Downloads" / "IPL.csv",
    ]

    csv_path = next((path for path in csv_candidates if path.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("Could not find IPL.csv in the project folder or Downloads.")

    return csv_path


def normalize_team_name(team_name: str) -> str:
    return TEAM_ALIASES.get(team_name, team_name)


def load_match_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(find_csv_path(), low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["batting_team"] = df["batting_team"].map(normalize_team_name)
    df["match_won_by"] = df["match_won_by"].map(lambda value: normalize_team_name(value) if pd.notna(value) else value)

    match_meta = (
        df.groupby("match_id")
        .agg(date=("date", "first"), winner=("match_won_by", "first"))
        .reset_index()
    )

    team_pairs = (
        df.groupby("match_id")["batting_team"]
        .unique()
        .apply(lambda teams: sorted(teams.tolist()))
        .reset_index(name="teams")
    )

    matches = match_meta.merge(team_pairs, on="match_id", how="inner")
    matches = matches[matches["teams"].map(len) == 2].copy()
    matches["team_1"] = matches["teams"].str[0]
    matches["team_2"] = matches["teams"].str[1]
    matches = matches.drop(columns=["teams"]).sort_values(["date", "match_id"]).reset_index(drop=True)

    all_teams = sorted(set(matches["team_1"]).union(matches["team_2"]))
    return matches, all_teams


def build_team_history(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    team_matches = matches[(matches["team_1"] == team) | (matches["team_2"] == team)].copy()
    team_matches["opponent"] = team_matches.apply(
        lambda row: row["team_2"] if row["team_1"] == team else row["team_1"],
        axis=1,
    )
    team_matches["won"] = team_matches["winner"] == team
    team_matches["result_known"] = team_matches["winner"].notna()
    return team_matches.sort_values(["date", "match_id"]).reset_index(drop=True)


def recent_win_rate(team_matches: pd.DataFrame, window: int) -> tuple[float, int, int]:
    recent = team_matches[team_matches["result_known"]].tail(window)
    games = len(recent)
    wins = int(recent["won"].sum())
    rate = wins / games if games else 0.5
    return rate, wins, games


def overall_win_rate(team_matches: pd.DataFrame) -> tuple[float, int, int]:
    completed = team_matches[team_matches["result_known"]]
    games = len(completed)
    wins = int(completed["won"].sum())
    rate = wins / games if games else 0.5
    return rate, wins, games


def head_to_head_rate(matches: pd.DataFrame, team_a: str, team_b: str, window: int) -> tuple[float, int, int]:
    h2h = matches[
        ((matches["team_1"] == team_a) & (matches["team_2"] == team_b))
        | ((matches["team_1"] == team_b) & (matches["team_2"] == team_a))
    ]
    h2h = h2h[h2h["winner"].notna()].sort_values(["date", "match_id"]).tail(window)
    games = len(h2h)
    wins = int((h2h["winner"] == team_a).sum())
    rate = wins / games if games else 0.5
    return rate, wins, games


def score_team(matches: pd.DataFrame, team: str, opponent: str) -> dict:
    team_history = build_team_history(matches, team)
    recent_rate, recent_wins, recent_games = recent_win_rate(team_history, RECENT_MATCHES)
    overall_rate, overall_wins, overall_games = overall_win_rate(team_history)
    h2h_rate, h2h_wins, h2h_games = head_to_head_rate(matches, team, opponent, HEAD_TO_HEAD_MATCHES)

    weighted_score = (0.6 * recent_rate) + (0.25 * overall_rate) + (0.15 * h2h_rate)

    return {
        "team": team,
        "recent_rate": recent_rate,
        "recent_wins": recent_wins,
        "recent_games": recent_games,
        "overall_rate": overall_rate,
        "overall_wins": overall_wins,
        "overall_games": overall_games,
        "h2h_rate": h2h_rate,
        "h2h_wins": h2h_wins,
        "h2h_games": h2h_games,
        "score": weighted_score,
    }


def explain_prediction(predicted: dict, other: dict) -> list[str]:
    reasons = []

    if predicted["recent_games"]:
        reasons.append(
            f"{predicted['team']} won {predicted['recent_wins']} of their last "
            f"{predicted['recent_games']} matches."
        )

    if predicted["recent_rate"] > other["recent_rate"]:
        reasons.append(
            f"Their recent form is stronger than {other['team']} "
            f"({predicted['recent_rate']:.0%} vs {other['recent_rate']:.0%})."
        )

    if predicted["h2h_games"] and predicted["h2h_rate"] > other["h2h_rate"]:
        reasons.append(
            f"They also have the better recent head-to-head record, winning "
            f"{predicted['h2h_wins']} of the last {predicted['h2h_games']} meetings."
        )

    if predicted["overall_rate"] > other["overall_rate"]:
        reasons.append(
            f"Their longer-term IPL win rate is also higher "
            f"({predicted['overall_rate']:.0%} vs {other['overall_rate']:.0%})."
        )

    return reasons[:3]


def resolve_team_name(user_input: str, teams: list[str]) -> str | None:
    cleaned = normalize_team_name(user_input.strip())
    if cleaned in teams:
        return cleaned

    lowered = cleaned.lower()
    exact_ignore_case = [team for team in teams if team.lower() == lowered]
    if exact_ignore_case:
        return exact_ignore_case[0]

    partial_matches = [team for team in teams if lowered in team.lower()]
    if len(partial_matches) == 1:
        return partial_matches[0]

    return None


def main() -> None:
    matches, teams = load_match_data()

    print("Available teams:")
    print(", ".join(teams))
    print()

    team_a_input = input("Enter Team 1: ")
    team_b_input = input("Enter Team 2: ")

    team_a = resolve_team_name(team_a_input, teams)
    team_b = resolve_team_name(team_b_input, teams)

    if team_a is None or team_b is None:
        print("One or both team names could not be matched. Please use a team name from the list above.")
        return

    if team_a == team_b:
        print("Please enter two different teams.")
        return

    team_a_score = score_team(matches, team_a, team_b)
    team_b_score = score_team(matches, team_b, team_a)

    total_score = team_a_score["score"] + team_b_score["score"]
    team_a_pct = (team_a_score["score"] / total_score) * 100 if total_score else 50.0
    team_b_pct = 100 - team_a_pct

    predicted, other = (team_a_score, team_b_score) if team_a_pct >= team_b_pct else (team_b_score, team_a_score)
    predicted_pct = max(team_a_pct, team_b_pct)
    other_pct = min(team_a_pct, team_b_pct)

    print()
    print("Match Prediction")
    print(f"{team_a}: {team_a_pct:.1f}% chance to win")
    print(f"{team_b}: {team_b_pct:.1f}% chance to win")
    print()
    print(f"Expected winner: {predicted['team']} ({predicted_pct:.1f}%)")
    print(f"Other side: {other['team']} ({other_pct:.1f}%)")
    print()
    print("Why this team is expected to win:")
    for reason in explain_prediction(predicted, other):
        print(f"- {reason}")


if __name__ == "__main__":
    main()
