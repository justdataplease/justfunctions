import collections
import concurrent.futures
import html
import inspect
import json
import logging
import math
import operator
import os
import pickle
import random
import re
import sys
import time
from copy import deepcopy
from datetime import date as date_type
from datetime import datetime, timedelta
from datetime import datetime as datetime_type
from functools import wraps
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import requests
import simplejson
from dateutil import relativedelta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

STOP_WORDS = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they',
              'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or',
              'who',
              'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
              'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she',
              'all',
              'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that',
              'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
              'only',
              'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how',
              'further', 'was', 'here', 'than']


def list_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def request_is_redirect(response):
    for x in response.history:
        if x.status_code == 302:
            return True
        else:
            return False


def website_regexp_validator(url):
    if url:
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return re.match(regex, url) is not None


def get_digit(text):
    try:
        text = text.replace(',', '')
        return float(re.findall(r'\d+(?:\.)?(?:\d+)?', text)[0])
    except Exception as exc:
        return None


def website_validator(url):
    try:
        response = requests.get(url, allow_redirects=True, timeout=10)
        if response.status_code <= 200:
            return url
        else:
            logger.info(response.status_code)
            return None
    except Exception as exc:
        logger.info(exc)
        if website_regexp_validator(url):
            return url
        else:
            return None


def get_abs_path(path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), path)


class CustomIntegrationError(Exception):
    """Base class for other exceptions"""

    def __init__(self, custom_error):
        self.status = custom_error.get('status')
        self.description = custom_error.get('error_description')
        self.description_verbose = custom_error.get('error_description_verbose')
        self.display_priority = custom_error.get('error_display_priority', 100)
        self.is_maya_integration_error = True


def escape_html_text(text):
    """escape strings for display in HTML"""
    rs = html.escape(text).replace(u'\n', u'<br />').replace(u'\t', u'&emsp;').replace(u'  ', u' &nbsp;')
    return rs if rs else ' '


def custom_integration_retry(attempts_dict={}, sleep_time_dict={}, default_attempts=5, default_sleep_time=6 * 15,
                             strategy='constant'):
    """
    Retry decorator

    @custom_integration_retry(attempts_dict={400: 3}, sleep_time_dict={400: 1}, default_attempts=5, default_sleep_time=1,
                          strategy='constant')
    def test_custom_integration_retry():
        raise CustomIntegrationError({'status': 400})

    test_custom_integration_retry()
    """

    def decorator(func):
        @wraps(func)
        def newfn(*args, **kwargs):
            previous_error_status = 12345678910
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    logger.info(f"{func.__name__} / {str(exc)}")
                    error_status = exc.status
                    retry_attempts = attempts_dict.get(error_status, default_attempts)

                    # If error status changed restart attempts
                    if previous_error_status != error_status:
                        attempt = 1
                    # if attempt > retry_attempts stop
                    if attempt > retry_attempts:
                        break
                    # Select strategy
                    if strategy == 'backoff':
                        retry_sleep_time = sleep_time_dict.get(error_status, default_sleep_time) * 2 ** attempt + random.uniform(0, 1)
                    else:
                        retry_sleep_time = sleep_time_dict.get(error_status, default_sleep_time)
                    logger.info(f'Retrying : attempt {attempt}/{retry_attempts}, {retry_sleep_time}s')
                    # Sleep
                    time.sleep(retry_sleep_time)
                    attempt += 1
                    previous_error_status = error_status
            return func(*args, **kwargs)

        return newfn

    return decorator


class UniqueDict(dict):
    def __setitem__(self, key, value):
        if key not in self:
            dict.__setitem__(self, key, value)
        else:
            raise KeyError("Key already exists")


def print_class_attr(attrs):
    rs = ', '.join("%s: %s" % item for item in attrs.items())
    print(rs)
    return rs


def custom_df_conditions(df, report_specs):
    ctr = report_specs['fields'].get('custom_transformations')
    if not ctr:
        return df

    ops = {
        '=': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '<': operator.lt,
        '=>': operator.le,
        '<=': operator.ge
    }
    field = ctr['field']
    condition = ctr['condition']
    value = int(ctr['value'])
    condition_func = ops[condition]

    return df[condition_func(pd.to_numeric(df[field]), value)].copy()


def fix_vname(word):
    if not word:
        return word
    return word.lower().replace(" ", "_")


def define_vname_column_name(x, x_init, dim_prefix):
    candidate = dim_prefix + x if x else dim_prefix[:-1]
    return x if '_x_id' in x_init else candidate


def define_vname_column_prefix(report_vname, report_vname_prefix):
    dim_prefix = ''
    if report_vname_prefix or report_vname[0:1] == '_':
        dim_prefix = report_vname_prefix if report_vname_prefix else report_vname[1:]
        dim_prefix += '_'
    return dim_prefix


def get_fields(columns, action=False, exclude_from=[], columns_prefix=False,
               type_map=False, return_excluded=False, keep_fields=[], ignore_fields=[]):
    rs = []
    rs_not = []
    for x in columns:
        alter_name = x.get("alias", x["name"])
        view_name = fix_vname(x.get("vname", alter_name))
        try:
            doc_text = x.get("doc", {}).get("text", "")
            doc_values = x.get("doc", {}).get("values", "")
        except Exception as exc:
            logger.error(f"get fields {exc}")
            doc_text = ""
            doc_values = ""

        if ((x.get('exclude_from', 'any') not in exclude_from) or (alter_name in keep_fields)) and (alter_name not in ignore_fields):
            if not action:
                # x.update({"alter_name": alter_name})
                j = deepcopy(x)
                j["alter_name"] = alter_name
                rs.append(j)
            elif action == 'get_names_types':
                rs.append((alter_name, type_map[x["type"]], view_name, doc_text, doc_values))
            elif action == 'get_names':
                rs.append(alter_name)
            elif action == 'get_names_view':
                rs.append(f'{alter_name} {columns_prefix}_{alter_name}')
            elif action == 'get_sql':
                rs.append(f'"{alter_name}" {type_map[x["type"]]}')
            else:
                raise Exception('get_fields : not supported...')
        else:
            if not action:
                # x.update({"alter_name": alter_name})
                j = deepcopy(x)
                j["alter_name"] = alter_name
                rs_not.append(j)
            elif action == 'get_names_types':
                rs_not.append((alter_name, type_map[x["type"]], view_name))
            elif action == 'get_names':
                rs_not.append(alter_name)
            elif action == 'get_names_view':
                rs_not.append(f'{alter_name} {columns_prefix}_{alter_name}')
            elif action == 'get_sql':
                rs_not.append(f'"{alter_name}" {type_map[x["type"]]}')
            else:
                raise Exception('get_fields : not supported...')

    if return_excluded:
        return rs_not
    return rs


def get_date_el(x):
    dt = str_to_date(x)
    return {'y': dt.year, 'm': dt.month, 'd': dt.day}


def date_to_unix(x, format='%Y-%m-%d'):
    return int(time.mktime(datetime.strptime(x, format).timetuple()))


def unix_to_date(x, format='%Y-%m-%d'):
    return datetime.fromtimestamp(float(x)).strftime(format)


def dict_rename_key(d, old_key, new_key):
    return {new_key if k == old_key else k: v for k, v in d.items()}


def get_schema_json(list_path):
    """
    get_schema_json(['services', 'google_ads', 'deps', 'schema.json'])
    :param list_path:
    :return:
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(project_root)
    dir = os.path.join(base_dir, *list_path)
    with open(dir) as f:
        return json.load(f)


def series_not_null(dt):
    dt_c = dt[dt.notnull() & dt.replace([np.inf, -np.inf], np.nan).notna() & (dt.astype(str).str.strip() != '')].copy()
    return dt_c.reset_index(drop=True)


def check_schema(df, report_specs, fill_missing_type=False, truncate_missing=False):
    common_fields = report_specs['report_schema']['columns']
    common_names = [f['name'] for f in common_fields]
    for x in df.columns:
        if x not in common_names:
            if fill_missing_type:
                logger.info(f'{x} added to schema {common_names}...')
                common_fields.append({'name': x, 'type': fill_missing_type})
            elif truncate_missing:
                logger.info(f'{x} dropped from schema...')
                df.drop(x, axis=1, inplace=True)
            else:
                raise Exception(f'{x} does not exist is schema {common_names}...')
    return common_fields


def get_unique_list_of_dicts(L):
    return [dict(s) for s in set(frozenset(d.items()) for d in L)]


def no_error(func, **kwargs):
    try:
        func(**kwargs)
        return True
    except:
        return False


def discover_series_type(s):
    # df = pd.DataFrame({'a': [100, 3, 4], 'b': [20.1, 2.3, 45.3], 'c': [None, 'a', 1.00],
    #                    'd': ['27/05/2001', '1999-01-01', '25/09/1998'], 'e': [True, False, True]})
    # s = df['a']

    s_clean = series_not_null(s)
    t = s_clean.dtype
    # Get type from dtype
    if issubclass(t.type, np.floating) or issubclass(t.type, np.integer) or issubclass(t.type, np.int64):
        sqltype = 'NUMERIC'
    elif no_error(pd.to_numeric, arg=s_clean, errors='raise'):
        sqltype = 'NUMERIC'
    elif issubclass(t.type, np.datetime64) or no_error(pd.to_datetime, arg=s_clean, errors='raise', exact=False):
        try:
            check_condition = pd.to_datetime(s_clean, errors='coerce', exact=False).dt.strftime(
                '%H:%M:%S') == '00:00:00'
            sqltype = 'DATE' if (check_condition[0] and check_condition.all()) else 'TIMESTAMP'
        except Exception as exc:
            sqltype = 'TIMESTAMP'
    elif issubclass(t.type, np.bool_):
        sqltype = 'BOOLEAN'
    elif issubclass(t.type, list):
        sqltype = 'LIST_AS_STR'
    else:
        sqltype = 'STRING'

    return sqltype


def discover_series_type_by_sample(s):
    s_clean = series_not_null(s)
    counter = 0
    sqltype_solutions = []
    while counter <= 3 and counter <= len(s_clean) - 1:
        sampl = s_clean[counter]
        try:
            float(sampl)
            sqltype_solutions.append('NUMERIC')
        except Exception as exc:
            if isinstance(sampl, str):
                sqltype_solutions.append('STRING')
            elif isinstance(sampl, date_type):
                sqltype_solutions.append('DATE')
            elif isinstance(sampl, datetime_type):
                sqltype_solutions.append('TIMESTAMP')
            elif isinstance(sampl, bool):
                sqltype_solutions.append('BOOLEAN')
            elif isinstance(sampl, list):
                sqltype_solutions.append('ARRAY_AS_STR')
            elif isinstance(sampl, dict):
                sqltype_solutions.append('DICT_AS_STR')
            else:
                logger.error(type(sampl))
        finally:
            counter += 1

    # Decide sample type
    logger.info(sqltype_solutions)
    if len(set(sqltype_solutions)) == 1:
        return sqltype_solutions[0]
    else:
        return 'STRING'

    return sqltype


def discover_df_types(df, by_sample=False):
    column_types = {}
    for k in df.columns:
        if by_sample:
            column_types[k] = {'type': discover_series_type_by_sample(df[k])}
        else:
            column_types[k] = {'type': discover_series_type(df[k])}
    return column_types


def discover_df_types_v2(df, by_sample=False):
    column_types = []
    for k in df.columns:
        if by_sample:
            column_types.append({'name': k, 'type': discover_series_type_by_sample(df[k])})
        else:
            column_types.append({'name': k, 'type': discover_series_type(df[k])})
    return column_types


def write_pickle(data, filename):
    pickle.dump(data, filename)


def open_pickle(filename):
    data = []
    with open(filename, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return data


def normalize_df_headers(df):
    """
    Normalize df colnames
    :param df:
    :param upper:
    :return:
    """

    colnames = list(df)

    df.columns = normalize_headers(k=colnames)
    logger.info(f'headers normalized : {df.columns}')
    return


def remove_last_symbol(iterable, symbol='_'):
    if iterable[-1] == symbol:
        return iterable[:len(iterable) - 1]
    else:
        return iterable


def remove_first_symbol(iterable, symbol='_'):
    if iterable[0] == symbol:
        return iterable[1:]
    else:
        return iterable


def normalize_headers(k):
    def normalize_item(x, bad_symbols, snake_case):
        k0 = snake_case.sub(r'_\1', x).lower()
        # Remove special symbols
        k1 = bad_symbols.sub('_', k0)
        # Strip & Replace Spaces
        k2 = re.sub('\s+', '_', k1.strip())
        # Replace multiple underscores
        k3 = re.sub('_+', '_', k2)
        # Replace last letter if symbol
        k3 = remove_last_symbol(k3, "_")
        k3 = remove_first_symbol(k3, "_")

        return k3

    """
    Fixes headers format
    :param list:
    """
    bad_symbols = re.compile(r'[^\w\s]+', flags=re.IGNORECASE)
    snake_case = re.compile(r'((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')

    if isinstance(k, list):
        rs = []
        for x in k:
            rs.append(normalize_item(x, bad_symbols, snake_case))
        return rs
    elif isinstance(k, str):
        return normalize_item(k, bad_symbols, snake_case)
    else:
        logger.warning(f'{k} not supported...')


def list_to_string(x):
    try:
        if not x:
            return ''
        else:
            return ','.join(map(str, x))
    except Exception:
        pass


def flatten(d, parent_key='', sep='_', list_max_elements=0):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            counter = 0
            for index, elem in enumerate(v):
                if isinstance(elem, collections.MutableMapping):
                    items.extend(flatten(elem, f"{new_key}_{counter}", sep=sep).items())
                    counter += 1
                    if counter > list_max_elements:
                        break
                else:
                    items.append((new_key, elem))
        else:
            items.append((new_key, v))
    return dict(items)


def list_to_str(lst):
    """
    List to string
    :return:
    """
    return ', '.join(map(lambda x: "'" + str(x) + "'", lst))


def stringify_list(lst):
    """
    List to string
    :return:
    """
    if not lst:
        return None
    return simplejson.dumps(lst, ignore_nan=True, default=str)


def remove_special_symbols(str):
    """
    Remove special symbols
    :param x:
    """
    pattern = r"[^\w\s]+"
    return re.sub(pattern, "", str.lower(), flags=re.IGNORECASE)


def remove_special_symbol_chars(str, sep=""):
    pattern = r"[\u00E2\u20AC×<>;:_¿§«»ω⊙¤°℃℉€¥£¢¡®©™`~!@#$%^&\*\(\)\|\+\–\-\=\?\'\’\",\.\{\}\[\]\\\/]+"
    return re.sub(pattern, sep, str, flags=re.IGNORECASE)


def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


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


def round_up_to_even(f):
    if f == 1:
        return 1
    return math.ceil(f / 2.) * 2


def dates_sequence_sublist_v2(start_date, end_date, step=30, completed_months=False):
    if completed_months:
        dates = dates_sequence_start_end_of_month(start_date, end_date)
        step = round_up_to_even(step)
    elif step == 0:
        step = 1
        logger.error('dates_sequence_sublist : setting step to 1...')
    else:
        dates = dates_sequence(start_date, end_date)
    dates_all = [dates[i:i + step] for i in range(0, len(dates), step)]
    dates_end = [[x[-0], x[-1]] for x in dates_all]
    return dates_end


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


def accepts(*types):
    def check_accepts(f):
        assert len(types) == f.__code__.co_argcount

        def new_f(*args, **kwds):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                    "arg %r does not match %s" % (a, t)
            return f(*args, **kwds)

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts


def parallel_jobs(items, func, var1, connections=5):
    with concurrent.futures.ProcessPoolExecutor(max_workers=connections) as executor:
        task_queue = [executor.submit(func, item, var1) for item in items]
        for task in concurrent.futures.as_completed(task_queue):
            try:
                task.result()
            except:
                yield task.result()


def run_concurrent_tasks(func, items, connections=5, *args):
    with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as executor:
        task_queue = [executor.submit(func, item, *args) for item in items]
        for task in concurrent.futures.as_completed(task_queue):
            try:
                logger.info(task.result())
            except Exception as exc:
                logger.info(exc)


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


def match_case(x, pattern, return_result=False, failcase=False):
    """
    Match pattern
    :param x:
    :param pattern:
    :param failcase:
    :param return_result:
    :return:
    """

    if not x:
        return False

    m_case = re.search(pattern, x, re.IGNORECASE)

    if not m_case:
        result = failcase
    else:
        if return_result:
            try:
                result = re.search(pattern, x, re.IGNORECASE).groups(0)[0]
            except Exception as exc:
                logger.error('match_case : ' + str(exc))
                result = failcase
        else:
            result = True
    return result


def match_all_case(x, pattern, failcase=[]):
    """
    Match pattern
    :param x:
    :param pattern:
    :param failcase:
    :return:
    """

    if not x:
        return False

    m_case = re.findall(pattern, x, re.IGNORECASE)

    if not m_case:
        result = failcase
    else:
        result = m_case
    return result


def isNaN(num):
    return num != num


def clear_input_symbols(x, ret=None):
    """
    Clear inpute symbols
    :param x:
    :return:
    """
    if not x or isNaN(x):
        return ret

    try:
        rs = re.sub(r'''\n|\r|\\r|\\n''', '', x)
    except Exception as exc:
        logger.error('clear_input_symbols : ' + str(exc))
        rs = ret
    finally:
        return rs


def remove_url_tail(x):
    """
    Remove url tail after ?
    :param x:
    :return:
    """
    return re.sub(r'\?.+|\#.+', '', x)


def decode_uri(url):
    """
    Decode utf8
    :param url:
    :return:
    """
    if isinstance(url, str):
        try:
            url = unquote(url)
        except Exception as exc:
            logger.error('decode_uri : ' + str(exc))
            url = ''
        return url
    else:
        return ''


def clean_url_link(url, no_domain=True):
    """
    Clean url link
    :param url:
    :param no_domain:
    :return:
    """
    if (not url) | (url in ['-', '--', '', ' ', ' --', '(not set)', 'undefined', '(undefined)']) | (
            not isinstance(url, str)):
        url = ''
        return url

    url = decode_uri(url)
    if no_domain:
        url = remove_domain(url)

    return url


def remove_elem(input, words):
    return list(filter(lambda x: x not in words, input))


def remove_domain(x):
    """
    Remove domain
    :param x:
    :return:
    """
    try:
        result = re.sub(r'(^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?(?:[^:\/?\n]+))', '', x, flags=re.IGNORECASE)
    except Exception as exc:
        logger.error('remove_domain : ' + str(exc))
        result = ''
    return result


def extract_email_domain(email):
    if (not email) | (email in ['-', '--', '', ' ', ' --']) | (not isinstance(email, str)):
        domain = ''
        return domain

    try:
        domain = re.search(r'@(.+)', email, flags=re.IGNORECASE).groups(0)[0]
    except Exception as exc:
        # logger.error('extract_email_domain : ' + str(exc))
        domain = ''
    return domain


def extract_email_subdomain(email):
    if (not email) | (email in ['-', '--', '', ' ', ' --']) | (not isinstance(email, str)):
        subdomain = ''
        return subdomain

    try:
        subdomain = re.search(r"@(.+)\.[\w]+$", email, flags=re.IGNORECASE).groups(0)[0]

    except Exception as exc:
        # logger.error('extract_email_subdomain : ' + str(exc))
        subdomain = ''
    return subdomain


def extract_domain(url):
    """
    Extract domain
    :param url:
    :return:
    """
    if (not url) | (url in ['-', '--', '', ' ', ' --']) | (not isinstance(url, str)):
        domain = ''
        return domain

    try:
        domain = re.search(r'^(?:https?:\/\/)?(?:[^@\/\n]+@)?((?:www\.)?([^:\/?\n]+))', url,
                           flags=re.IGNORECASE).groups(0)[0]
    except Exception as exc:
        # logger.error('extract_domain : ' + str(exc))
        domain = ''
    return domain


def df_split(df, n=1000):
    for i in range(0, df.shape[0], n):
        yield df[i:i + n]


def df_split_by_field(df, field='date', backfill=False, ignore=False):
    """
        import pandas as pd

        df = pd.DataFrame(
            {
                "Company": [
                    "Samsung", "Samsung", "Samsung", "Samsung", "Samsung", "LG", "LG", "LG", "LG", "LG", "Sony", "Sony", "Sony",
                    "Sony", "Sony",
                ],
                "date": [
                    "10/9/2015", "10/9/2015", "10/9/2017", "10/10/2017", "10/10/2017", "10/10/2018", "10/9/2018", "10/9/2018",
                    "10/9/2018", "10/10/2016", "10/10/2016", "10/10/2016", "10/10/2019", "10/10/2019", "10/10/2019",
                ],
                "Country": [
                    "India", "India", "USA", "France", "India", "India", "Germany", "USA", "Brazil", "Brazil", "India",
                    "Germany", "India", "India", "Brazil",
                ],
                "Sells": [15, 81, 29, 33, 21, 42, 67, 35, 2, 34, 21, 50, 10, 26, 53],
            }
        )


        for x in df_split_by_field(df):
            for y in df_split_by_size(x):
                print(y)
    :param df:
    :param filed:
    :return:
    """
    if ignore:
        yield {'field': '', 'data': df}
    else:
        if backfill:
            df.sort_values(field, ascending=False, inplace=True)
        else:
            df.sort_values(field, ascending=True, inplace=True)

        for x, y in df.groupby(field, as_index=False, sort=False):
            yield {'field': x, 'data': pd.DataFrame(y)}


def df_split_by_size(df, mb=1000):
    """
    Generator function that splits dataframe into chunks based on megabytes
    Example:
    df = pd.DataFrame(np.random.randint(0, 100, size=(100000000, 4)), columns=list('ABCD'))
    sys.getsizeof(df)/1000000

    rs = []
    for rdf in df_split_by_size(df, mb=10):
        rs.append(rdf)

    adfs = pd.concat(rs)
    df.equals(adfs)

    :param df: pandas dataframe
    :param mb: megabytes
    :return: pandas dataframe chunk
    """
    bmb = 1000000
    size = sys.getsizeof(df)
    # size = df.memory_usage(deep=True).sum()
    if size / bmb >= mb:
        logger.info(f'size exceeded {mb}mb')
        ndfs = np.array_split(df, 2)
        for ndf in ndfs:
            yield from df_split_by_size(ndf, mb=mb)
    else:
        yield df


def get_dir_files(path, pattern):
    files = []
    for file in os.listdir(path):
        if file.endswith(pattern):
            files.append(file)
    return files


def elapsed_since(start):
    # return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    elapsed = time.time() - start
    if elapsed < 1:
        return str(round(elapsed * 1000, 2)) + "ms"
    if elapsed < 60:
        return str(round(elapsed, 2)) + "s"
    if elapsed < 3600:
        return str(round(elapsed / 60, 2)) + "min"
    else:
        return str(round(elapsed / 3600, 2)) + "hrs"


def get_process_memory():
    import psutil
    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    return mi.rss, mi.vms


def format_bytes(bytes):
    if abs(bytes) < 1000:
        return str(bytes) + "B"
    elif abs(bytes) < 1e6:
        return str(round(bytes / 1e3, 2)) + "kB"
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"


def memory_profile(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        rss_before, vms_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        rss_after, vms_after = get_process_memory()
        print("Profiling: {:>20}  RSS: {:>8} | VMS: {:>8} | time: {:>8}"
              .format("<" + func.__name__ + ">",
                      format_bytes(rss_after - rss_before),
                      format_bytes(vms_after - vms_before),
                      elapsed_time))
        return result

    if inspect.isfunction(func):
        return wrapper
    elif inspect.ismethod(func):
        return wrapper(*args, **kwargs)


def check_position(a, pos, ret=None):
    try:
        r = a[pos]
    except IndexError:
        r = ret
    except KeyError:
        r = ret
    return r


def tryget(x, name, failcase=None):
    if x is None:
        return failcase
    try:
        if x[name]:
            return x[name]
        else:
            return failcase
    except (KeyError, AttributeError):
        return failcase


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def find_country_from_domain(d, return_code=True):
    domain = extract_domain(d)
    l = {'ac': 'Ascension Island (United Kingdom)',
         'ad': 'Andorra',
         'ae': 'United Arab Emirates',
         'af': 'Afghanistan',
         'ag': 'Antigua and Barbuda',
         'ai': 'Anguilla (United Kingdom)',
         'al': 'Albania',
         'am': 'Armenia',
         'ao': 'Angola',
         'aq': 'Antarctica',
         'ar': 'Argentina',
         'as': 'American Samoa (United States)',
         'at': 'Austria',
         'au': 'Australia',
         'aw': 'Aruba (Kingdom of the Netherlands)',
         'ax': 'Aland (Finland)',
         'az': 'Azerbaijan',
         'ba': 'Bosnia and Herzegovina',
         'bb': 'Barbados',
         'bd': 'Bangladesh',
         'be': 'Belgium',
         'bf': 'Burkina Faso',
         'bg': 'Bulgaria',
         'bh': 'Bahrain',
         'bi': 'Burundi',
         'bj': 'Benin',
         'bm': 'Bermuda (United Kingdom)',
         'bn': 'Brunei',
         'bo': 'Bolivia',
         'bq': 'Caribbean Netherlands ( Bonaire,  Saba, and  Sint Eustatius)',
         'br': 'Brazil',
         'bs': 'Bahamas',
         'bt': 'Bhutan',
         'bw': 'Botswana',
         'by': 'Belarus',
         'bz': 'Belize',
         'ca': 'Canada',
         'cc': 'Cocos (Keeling) Islands (Australia)',
         'cd': 'Democratic Republic of the Congo',
         'cf': 'Central African Republic',
         'cg': 'Republic of the Congo',
         'ch': 'Switzerland',
         'ci': 'Ivory Coast',
         'ck': 'Cook Islands',
         'cl': 'Chile',
         'cm': 'Cameroon',
         'cn': "People's Republic of China",
         'co': 'Colombia',
         'cr': 'Costa Rica',
         'cu': 'Cuba',
         'cv': 'Cape Verde',
         'cw': 'Curaçao (Kingdom of the Netherlands)',
         'cx': 'Christmas Island',
         'cy': 'Cyprus',
         'cz': 'Czech Republic',
         'de': 'Germany',
         'dj': 'Djibouti',
         'dk': 'Denmark',
         'dm': 'Dominica',
         'do': 'Dominican Republic',
         'dz': 'Algeria',
         'ec': 'Ecuador',
         'ee': 'Estonia',
         'eg': 'Egypt',
         'eh': 'Western Sahara',
         'er': 'Eritrea',
         'es': 'Spain',
         'et': 'Ethiopia',
         'eu': 'European Union',
         'fi': 'Finland',
         'fj': 'Fiji',
         'fk': 'Falkland Islands (United Kingdom)',
         'fm': 'Federated States of Micronesia',
         'fo': 'Faroe Islands (Kingdom of Denmark)',
         'fr': 'France',
         'ga': 'Gabon',
         'gd': 'Grenada',
         'ge': 'Georgia',
         'gf': 'French Guiana (France)',
         'gg': 'Guernsey (United Kingdom)',
         'gh': 'Ghana',
         'gi': 'Gibraltar (United Kingdom)',
         'gl': 'Greenland (Kingdom of Denmark)',
         'gm': 'The Gambia',
         'gn': 'Guinea',
         'gp': 'Guadeloupe (France)',
         'gq': 'Equatorial Guinea',
         'gr': 'Greece',
         'gs': 'South Georgia and the South Sandwich Islands (United Kingdom)',
         'gt': 'Guatemala',
         'gu': 'Guam (United States)',
         'gw': 'Guinea-Bissau',
         'gy': 'Guyana',
         'hk': 'Hong Kong',
         'hm': 'Heard Island and McDonald Islands',
         'hn': 'Honduras',
         'hr': 'Croatia',
         'ht': 'Haiti',
         'hu': 'Hungary',
         'id': 'Indonesia',
         'ie': 'Ireland',
         'il': 'Israel',
         'im': 'Isle of Man (United Kingdom)',
         'in': 'India',
         'io': 'British Indian Ocean Territory (United Kingdom)',
         'iq': 'Iraq',
         'ir': 'Iran',
         'is': 'Iceland',
         'it': 'Italy',
         'je': 'Jersey (United Kingdom)',
         'jm': 'Jamaica',
         'jo': 'Jordan',
         'jp': 'Japan',
         'ke': 'Kenya',
         'kg': 'Kyrgyzstan',
         'kh': 'Cambodia',
         'ki': 'Kiribati',
         'km': 'Comoros',
         'kn': 'Saint Kitts and Nevis',
         'kp': 'North Korea',
         'kr': 'South Korea',
         'kw': 'Kuwait',
         'ky': 'Cayman Islands (United Kingdom)',
         'kz': 'Kazakhstan',
         'la': 'Laos',
         'lb': 'Lebanon',
         'lc': 'Saint Lucia',
         'li': 'Liechtenstein',
         'lk': 'Sri Lanka',
         'lr': 'Liberia',
         'ls': 'Lesotho',
         'lt': 'Lithuania',
         'lu': 'Luxembourg',
         'lv': 'Latvia',
         'ly': 'Libya',
         'ma': 'Morocco',
         'mc': 'Monaco',
         'md': 'Moldova',
         'me': 'Montenegro',
         'mg': 'Madagascar',
         'mh': 'Marshall Islands',
         'mk': 'North Macedonia',
         'ml': 'Mali',
         'mm': 'Myanmar',
         'mn': 'Mongolia',
         'mo': 'Macau',
         'mp': 'Northern Mariana Islands (United States)',
         'mq': 'Martinique (France)',
         'mr': 'Mauritania',
         'ms': 'Montserrat (United Kingdom)',
         'mt': 'Malta',
         'mu': 'Mauritius',
         'mv': 'Maldives',
         'mw': 'Malawi',
         'mx': 'Mexico',
         'my': 'Malaysia',
         'mz': 'Mozambique',
         'na': 'Namibia',
         'nc': 'New Caledonia (France)',
         'ne': 'Niger',
         'nf': 'Norfolk Island',
         'ng': 'Nigeria',
         'ni': 'Nicaragua',
         'nl': 'Netherlands',
         'no': 'Norway',
         'np': 'Nepal',
         'nr': 'Nauru',
         'nu': 'Niue',
         'nz': 'New Zealand',
         'om': 'Oman',
         'pa': 'Panama',
         'pe': 'Peru',
         'pf': 'French Polynesia (France)',
         'pg': 'Papua New Guinea',
         'ph': 'Philippines',
         'pk': 'Pakistan',
         'pl': 'Poland',
         'pm': 'Saint-Pierre and Miquelon (France)',
         'pn': 'Pitcairn Islands (United Kingdom)',
         'pr': 'Puerto Rico (United States)',
         'ps': 'Palestine[56]',
         'pt': 'Portugal',
         'pw': 'Palau',
         'py': 'Paraguay',
         'qa': 'Qatar',
         're': 'Réunion (France)',
         'ro': 'Romania',
         'rs': 'Serbia',
         'ru': 'Russia',
         'rw': 'Rwanda',
         'sa': 'Saudi Arabia',
         'sb': 'Solomon Islands',
         'sc': 'Seychelles',
         'sd': 'Sudan',
         'se': 'Sweden',
         'sg': 'Singapore',
         'sh': 'Saint Helena, Ascension and Tristan da Cunha (United Kingdom)',
         'si': 'Slovenia',
         'sk': 'Slovakia',
         'sl': 'Sierra Leone',
         'sm': 'San Marino',
         'sn': 'Senegal',
         'so': 'Somalia',
         'sr': 'Suriname',
         'ss': 'South Sudan',
         'st': 'São Tomé and Príncipe',
         'su': 'Soviet Union',
         'sv': 'El Salvador',
         'sx': 'Sint Maarten (Kingdom of the Netherlands)',
         'sy': 'Syria',
         'sz': 'Eswatini',
         'tc': 'Turks and Caicos Islands (United Kingdom)',
         'td': 'Chad',
         'tf': 'French Southern and Antarctic Lands',
         'tg': 'Togo',
         'th': 'Thailand',
         'tj': 'Tajikistan',
         'tk': 'Tokelau',
         'tl': 'East Timor',
         'tm': 'Turkmenistan',
         'tn': 'Tunisia',
         'to': 'Tonga',
         'tr': 'Turkey',
         'tt': 'Trinidad and Tobago',
         'tv': 'Tuvalu',
         'tw': 'Taiwan',
         'tz': 'Tanzania',
         'ua': 'Ukraine',
         'ug': 'Uganda',
         'uk': 'United Kingdom',
         'us': 'United States of America',
         'com': 'United States of America',
         'uy': 'Uruguay',
         'uz': 'Uzbekistan',
         'va': 'Vatican City',
         'vc': 'Saint Vincent and the Grenadines',
         've': 'Venezuela',
         'vg': 'British Virgin Islands (United Kingdom)',
         'vi': 'United States Virgin Islands (United States)',
         'vn': 'Vietnam',
         'vu': 'Vanuatu',
         'wf': 'Wallis and Futuna',
         'ws': 'Samoa',
         'ye': 'Yemen',
         'yt': 'Mayotte',
         'za': 'South Africa',
         'zm': 'Zambia',
         'zw': 'Zimbabwe'}

    for x in l.keys():
        if match_case(domain, pattern=fr"\.{x}$") or match_case(domain, pattern=fr"^{x}\."):
            if return_code:
                return x
            else:
                return l[x]
    if return_code:
        return 'us'
    else:
        return l['us']


def get_ip_details(ip_address):
    import ipinfo
    access_token = ''
    handler = ipinfo.getHandler(access_token)
    details = handler.getDetails(ip_address)
    return details.all


def ip_from_domain(d):
    import socket
    return socket.gethostbyname(d)


def get_domain_hosted_details(d):
    d = extract_domain(d)
    ip_address = ip_from_domain(d)
    return get_ip_details(ip_address)


def generate_docker_command(project_name, steps=[], reports_to_run=[], reports_to_rest=[],
                            routines_to_run=[], routine_actions_to_run=[]):
    print('gcloud beta compute ssh --zone "europe-west3-a" "maya-etl-1" --project "bi-analytics-253109"')
    print('cd /home/maya/cleric &&')

    if project_name == 'all':
        base = f'docker-compose run worker run -workflow run_etl -all-schedules'
    else:
        base = f'docker-compose run --rm worker run -schedule {project_name}'

    if steps:
        base += f" -- --steps={','.join(steps)}"

    if reports_to_run:
        base += f" --reports-to-run={','.join(reports_to_run)}"

    if reports_to_rest:
        base += f" --reports-to-reset={','.join(reports_to_rest)}"

    if routines_to_run:
        base += f" --routines-to-run={','.join(routines_to_run)}"

    if routine_actions_to_run:
        base += f" --routine-actions-to-run={','.join(routine_actions_to_run)}"

    print(base)


def get_demo_vocabulary():
    p = Path(__file__).with_name('demo_vocabulary_small.json')
    demo_voc = json.load(open(p))
    return demo_voc


def randomize_series(s, ttype, params):
    random_factor = params.get('random_factor')

    if random_factor == 1:
        return s

    np.random.seed(1234)
    if ttype == "numeric":
        if random_factor:
            multiplier = random_factor
        else:
            multiplier = np.random.uniform(low=0.2, high=1.5)

        return s.apply(lambda x: int(x * multiplier)).copy()
    elif ttype == "string":
        def complete_rule(rule):
            return "|".join([f"\w*{x}\w*" for x in rule.split("|")])

        rules = {"cryptotrade": "polfi|pollfi|polefi|plfi|pllfi|poll fi|pol fi|poolf|pooo|polif",
                 "trading": "research",
                 "investor ": "customer",
                 "coin": "response",
                 "cardano": "focus group|focus|segment",
                 "coinbase wallet": "google poll ",
                 "coinbase": "google|gogle|voting",
                 "marketplace": "branded",
                 "blockchain plan": " plan ",
                 "blockchain": " poll ",
                 "market": "market|markt|form",
                 "crypto": "surv|srve|quest|qust|chart|visual|dashb|descr|brand",
                 "platform": "satisfa|nps|feedback",
                 "profit": "score",
                 "loss": "error|bias",
                 "decentralized": "online|open",
                 "quantitative": "quant|qualit",
                 "money": "monney|money",
                 "cryptocom": "monkey",
                 "kraken": "agoda",
                 "etherium": "typeform",
                 "nfts": "example|exper|sampl|smpl",
                 "reinvest": "calcul|clcu|scale|conduct",
                 "bitcoin": "respond|docs",
                 "increase": "promote",
                 "earnings": "proba|distri|dstri",
                 "news": "aware",
                 "impact": "correl|corel|statistic|confid",
                 "bull": "statistic|pump and dump",
                 "rank": "rank|interval|intrval",
                 "cryptowallet": "answer|answ",
                 "ether": "explorat|resource|explan",
                 "analysis": "analy|anly",
                 "ethereum": "panel|blog",
                 "probability": "rating",
                 "metaverse": "design",
                 "helium": "likert",
                 "invest": "consume"
                 }

        for i in list(rules):
            regex_pat = re.compile(r"{rules}".format(rules=complete_rule(rules[i])), flags=re.IGNORECASE)
            s = s.str.replace(regex_pat, i, regex=True)

        return s


def create_schema_from_df(df):
    schema = []
    for x in list(df):
        if match_case(x, pattern='time|_at|date|last_modified'):
            type = "DATE"
        else:
            type = "STRING"
        schema.append({"name": f"{x}", "type": type})
    return json.dumps(schema, indent=4)


def add_url_schema(url):
    from urllib.parse import urlparse, ParseResult
    p = urlparse(url, 'https')
    netloc = p.netloc or p.path
    path = p.path if p.netloc else ''
    if not netloc.startswith('www.'):
        netloc = 'www.' + netloc

    p = ParseResult('https', netloc, path, *p[3:])
    return p.geturl()


def extract_url_netlocs(urls):
    rs = []
    for x in urls:
        rs.append(urlparse(x).netloc)
    return rs


def isinstance_df(input_df):
    return input_df if isinstance(input_df, pd.DataFrame) else pd.DataFrame()
