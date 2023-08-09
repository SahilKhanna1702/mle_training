def test_package_installation():
    try:
        import HousingPricePredictions
    except Exception as e:
        assert False
        f"Package is not installed properly {e}"
