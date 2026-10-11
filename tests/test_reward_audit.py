from scripts.reward_audit import RAW_FIELDS, summarise


def test_reward_audit_reports_frequency_mean_and_total():
    rows = [{field: 0.0 for field in RAW_FIELDS} for _ in range(2)]
    rows[0]["spell_prediction"] = 0.5
    rows[0]["kite_quality"] = 1.0
    rows[1]["kite_quality"] = 0.25

    result = summarise(rows)
    assert result["spell_prediction"] == {
        "mean_per_episode": 0.25, "nonzero_episodes": 1, "total": 0.5}
    assert result["kite_quality"] == {
        "mean_per_episode": 0.625, "nonzero_episodes": 2, "total": 1.25}
