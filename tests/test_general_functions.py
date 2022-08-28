from doingscience.general_functions import check_position


def test_check_position():
    x = ['test']
    assert check_position(x, pos=0, ret=None) == 'test'
    assert check_position(x, pos=1, ret='na') == 'na'
