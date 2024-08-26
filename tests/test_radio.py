from normative_diversity.RADio.distributions import harmonic_number


def test_harmonic_number():
    assert harmonic_number(1) == 1.0022156649015328
    assert harmonic_number(5) == 2.2833335773356334
    assert harmonic_number(10) == 2.9289682578955785
