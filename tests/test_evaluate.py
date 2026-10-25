import pytest
from evaluate import make_salty_profiles, TIERS
from ranking import recommend, scaled_table, SODIUM_COLUMN


@pytest.fixture(scope="module")
def salty():
    """Build the 200 salty users once and share them across these tests"""
    return make_salty_profiles()


def test_salty_profiles_have_100_users_per_tier(salty):
    assert len(salty) == 200
    for tier in TIERS:
        assert sum(p["sodium_limit"] == tier for _, p in salty) == 100


def test_each_salty_user_has_two_goals(salty):
    for _, p in salty:
        assert (p["weights"] == 3.0).sum() == 2


def test_sodium_is_never_a_goal(salty):
    # Goals and the sodium rule stay separate: sodium keeps the default median target and weight
    median = scaled_table[SODIUM_COLUMN].median()
    for _, p in salty:
        assert p["weights"][SODIUM_COLUMN] == 1.0
        assert p["u"][SODIUM_COLUMN] == median


def test_same_users_every_time(salty):
    again = make_salty_profiles()
    for (name_a, a), (name_b, b) in zip(salty, again):
        assert name_a == name_b
        assert a["u"].equals(b["u"])


def test_hard_filter_is_safe_for_every_salty_user(salty):
    for _, p in salty:
        result = recommend(p, 10)
        assert len(result) == 10
        assert (result[SODIUM_COLUMN] <= p["sodium_limit"] / 3).all()
