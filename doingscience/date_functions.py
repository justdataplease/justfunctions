import logging
from dateutil import relativedelta
from datetime import date as date_type, datetime, timedelta
from general_functions import round_up_to_even
from time import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_date_el(x):
    dt = str_to_date(x)
    return {'y': dt.year, 'm': dt.month, 'd': dt.day}


def date_to_unix(x, format='%Y-%m-%d'):
    return int(time.mktime(datetime.strptime(x, format).timetuple()))


def unix_to_date(x, format='%Y-%m-%d'):
    return datetime.fromtimestamp(float(x)).strftime(format)


def evaluate_dates(d1, d2, check, d1_format='%Y-%m-%d', d2_format='%Y-%m-%d', return_boolean=False):
    d1 = to_date(d1, d1_format)
    d2 = to_date(d2, d2_format)
    if check == 'vs_each_other':
        if d1 > d2:
            if return_boolean:
                return True
            raise Exception(f'wrong dates {d1}>{d2}')
    elif check == 'vs_today':
        tdy = today(to_date=True)
        if d2 > tdy:
            if return_boolean:
                return True
            raise Exception(f'wrong dates {d2}>{tdy}')


def substract_dates(d1, d2, d1_format='%Y-%m-%d', d2_format='%Y-%m-%d'):
    d1 = to_date(d1, d1_format)
    d2 = to_date(d2, d2_format)

    if d1 > d2:
        logger.error(f'{d1}>{d2}')
        return None

    difference = (d2 - d1)
    total_seconds = difference.total_seconds()

    return total_seconds


def el_to_date(y, m, d, format='%Y-%m-%d'):
    x = datetime(y, m, d)
    return date_to_str(x, format)


def str_to_date(date, format='%Y-%m-%d', safe=False):
    try:
        rs = datetime.strptime(date, format)
    except Exception as exc:
        if safe:
            return None
        rs = None
        logger.error(f'str_to_date : date {date} is not compatible / {exc}')
    return rs


def transform_date_str_format(date, format):
    return date_to_str(str_to_date(date), format=format)


def date_to_str(date, format='%Y-%m-%d'):
    return datetime.strftime(date, format)


def to_date(d, format='%Y-%m-%d'):
    if isinstance(d, datetime):
        d1 = d.date()  # start date
    elif isinstance(d, date_type):
        d1 = d  # start date
    elif isinstance(d, str):
        d1 = datetime.strptime(d, format).date()  # start date
    else:
        d1 = None
        logger.error(f'{d} cannot be converted to date')

    return d1


def add_days(date=None, num=None, date_format="%Y-%m-%d"):
    """
    Add days to date
    :param date:
    :param num:
    :param date_format:
    :return:
    """
    if not date:
        date = datetime.utcnow() + timedelta(days=-1)

    date = to_date(date)
    result_date = date + timedelta(days=num)
    today = datetime.utcnow()
    yesterday = today + timedelta(days=-1)
    return {"date": date.strftime(date_format), "result_date": result_date.strftime(date_format),
            "today": today.strftime(date_format), "yesterday": yesterday.strftime(date_format)}


def today(formatted=False, date_format="%Y-%m-%d %H:%M:%S", to_date=False):
    today = datetime.utcnow()
    if formatted:
        return today.strftime(date_format)
    else:
        if to_date:
            return today.date()
        else:
            return today


def start_of_span(x, span='year', input_format='%Y-%m-%d', export_format='%Y-%m-%d', formatted=False):
    if isinstance(x, date_type):
        d = x
    else:
        d = datetime.strptime(x, input_format)

    if span == 'day':
        rd = d.replace(hour=0, minute=0, second=0, microsecond=0)
    elif span == 'year':
        rd = d.replace(day=1, month=1)
    elif span == 'month':
        rd = d.replace(day=1)
    else:
        logger.error(f'start_of_span : {span} not supported')

    if formatted:
        return str(rd.strftime(export_format))
    else:
        return rd


def add_months(date, num, date_format="%Y-%m-%d", start_of_month=True, keep_datetime=False):
    """
    Add days to date
    :param date:
    :param num:
    :param date_format:
    :return:
    """
    if not date:
        date = datetime.utcnow() + timedelta(days=-1)
    result_date = date + relativedelta.relativedelta(months=num)
    if start_of_month:
        result_date = result_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    today = datetime.utcnow()
    if keep_datetime:
        return {"date": date, "result_date": result_date, "today": today}
    else:
        return {"date": date.strftime(date_format), "result_date": result_date.strftime(date_format),
                "today": today.strftime(date_format)}


def dates_sequence(start_date, end_date):
    seq = []
    d1 = to_date(start_date)  # start date
    d2 = to_date(end_date)  # end date
    delta = d2 - d1
    for i in range(delta.days + 1):
        m = (d1 + timedelta(days=i))
        seq.append(str(m.strftime("%Y-%m-%d")))
    return seq


def dates_sequence_start_end_of_month(start_date, end_date):
    seq = []
    d1 = to_date(start_date)  # start date
    d2 = to_date(end_date)  # end date
    d2_start_of_month = d2.replace(day=1)
    delta = d2 - d1
    for i in range(delta.days + 1):
        loop_date = d1 + timedelta(days=i)
        m = loop_date.replace(day=1)
        if loop_date >= d2_start_of_month:
            m2 = d2
        else:
            m2 = m + relativedelta.relativedelta(months=1) - timedelta(days=1)
        seq.append(m)
        seq.append(m2)

    seq = list(set(seq))
    seq.sort()
    return [str(x.strftime("%Y-%m-%d")) for x in seq]


def dates_sequence_start_of_month(start_date, end_date):
    seq = []
    d1 = to_date(start_date)  # start date
    d2 = to_date(end_date)  # end date
    delta = d2 - d1
    for i in range(delta.days + 1):
        m = (d1 + timedelta(days=i)).replace(day=1)
        seq.append(m)
    seq = list(set(seq))
    seq.sort()
    return [str(x.strftime("%Y-%m-%d")) for x in seq]


def dates_sequence_sublist(start_date, end_date, step=30, completed_months=False):
    if completed_months:
        dates = dates_sequence_start_end_of_month(start_date, end_date)
        step = round_up_to_even(step) * 2
    elif step == 0:
        step = 1
        logger.error('dates_sequence_sublist : setting step to 1...')
    else:
        dates = dates_sequence(start_date, end_date)

    dates_all = [dates[i:i + step] for i in range(0, len(dates), step)]
    dates_end = [[x[-0], x[-1]] for x in dates_all]
    return dates_end
