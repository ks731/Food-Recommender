from user_profiles import add_low_sodium, make_profile, PRESETS, scaled_table


def test_add_low_sodium_sets_sodium_target_and_weight():
    profile = make_profile(scaled_table, PRESETS["high_protein"], 1500)
    low = add_low_sodium(scaled_table, profile, pct=0.10, weight=3.0)
    assert low["u"]["Sodium, Na"] == scaled_table["Sodium, Na"].quantile(0.10)
    assert low["weights"]["Sodium, Na"] == 3.0


def test_add_low_sodium_leaves_other_nutrients_alone():
    profile = make_profile(scaled_table, PRESETS["high_protein"], 1500)
    low = add_low_sodium(scaled_table, profile)
    others = profile["u"].index.drop("Sodium, Na")
    assert (low["u"][others] == profile["u"][others]).all()
    assert (low["weights"][others] == profile["weights"][others]).all()
    assert low["sodium_limit"] == 1500


def test_add_low_sodium_does_not_change_original_profile():
    profile = make_profile(scaled_table, PRESETS["high_protein"], 1500)
    before = profile["u"]["Sodium, Na"], profile["weights"]["Sodium, Na"]
    add_low_sodium(scaled_table, profile)
    assert (profile["u"]["Sodium, Na"], profile["weights"]["Sodium, Na"]) == before
